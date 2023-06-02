from abc import ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence, Set, Collection, Iterable
from dataclasses import dataclass, field, is_dataclass, fields
from typing import TypeVar, Generic
from types import GenericAlias

from collections import defaultdict
from pathlib import Path
from xml.etree.ElementTree import ElementTree, Element
import itertools
import json
import logging

from auroratools.core.utils import ProgressTracker, transform_mapping_values, subclasses

_logger = logging.getLogger(__name__)


# ABSTRACT PARSING INTERFACE

KeyT = TypeVar("KeyT")
DataT = TypeVar("DataT")

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Parser(Generic[InputT, OutputT], metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def create(cls, something: InputT) -> OutputT:
        raise NotImplementedError()

    @classmethod
    def transform_index(cls, input_index: Mapping[KeyT, InputT],
                        progress: ProgressTracker | None = None) -> dict[KeyT, OutputT]:
        return transform_mapping_values(cls.create, input_index, progress, "TRANSFORM")


class AuroraParser(Generic[InputT, OutputT], Parser[InputT, OutputT], metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def load_aurora_content(cls, *sub_dirs: str | Path, base: str | Path = "",
                            progress: ProgressTracker | None = None) -> Mapping[str, OutputT]:
        """Two-stage process: 1. indexing, 2. parsing => progress tracker is called twice"""
        raise NotImplementedError()


# CONTENT FILE INDEXING AND PRE-PARSING

XMLNodes = Sequence[Element]
XMLIndex = dict[str, XMLNodes]


def _fuse_index(main_nodes: Mapping[str, Element], appendixes: Mapping[str, Collection[Element]]) -> XMLIndex:
    return {aurora_id: (main_node, *appendixes.get(aurora_id, ())) for aurora_id, main_node in main_nodes.items()}


def index_aurora_files(*content_dirs: str | Path, base: str | Path = "",
                       progress: ProgressTracker | None = None) -> XMLIndex:
    base = Path(base)
    files = sorted(file for path in content_dirs for file in (base / Path(path)).glob("**/*.xml"))
    main_nodes = dict[str, Element]()
    appendixes = defaultdict[str, list[Element]](list)
    for file in (progress(files, "INDEX") if progress else files):
        xml_root = ElementTree(file=file).getroot()
        for node in xml_root:
            if node.tag == "element":
                aurora_id = node.attrib["id"]
                if aurora_id in main_nodes:
                    _logger.warning("Duplicated main node for %s", aurora_id)
                else:
                    main_nodes[aurora_id] = node
            elif node.tag == "append":
                appendixes[node.attrib["id"]].append(node)
            # ignore node.tag == "info"
    return _fuse_index(main_nodes, appendixes)


@dataclass(frozen=True)
class PreParsing(AuroraParser[XMLNodes, "PreParsing"]):
    id: str
    kind: str
    name: str
    source: str
    flags: list[str] = field(default_factory=list)  # supports
    attrs: dict[str, str] = field(default_factory=dict)  # setters
    setters: dict[str, Element] = field(default_factory=dict)
    others: dict[str, list[Element]] = field(default_factory=lambda: defaultdict(list))

    SETTER_PARENT_TAGS = {"setters", "setter"}
    OTHER_ACCEPTED_TAGS = {"description", "sheet", "compendium", "prerequisite", "prerequisites",
                           "requirements", "rules", "spellcasting", "extract", "multiclass"}

    def _update_setters(self, setter_parent_node: Element):
        for setter in setter_parent_node.iterfind("set"):
            setter_name = setter.attrib["name"]
            if setter_name == "names" and "type" in setter.attrib:
                setter_name = setter.attrib["type"]
            if setter.text is None:
                _logger.info("Empty setter named '%s' in %s", setter_name, self.id)
            elif setter_name in self.attrs:
                _logger.warning("Duplicated setter name '%s' in %s", setter_name, self.id)
            else:
                self.attrs[setter_name] = (setter.text or "").strip()
                self.setters[setter_name] = setter

    @classmethod
    def create(cls, nodes: XMLNodes) -> "PreParsing":
        attrs = nodes[0].attrib
        self = PreParsing(attrs["id"], attrs["type"], attrs["name"], attrs["source"])
        for node in nodes:
            for child_node in node:
                child_tag = child_node.tag
                if child_tag == "supports":
                    for flag in (child_node.text or "").split(","):
                        self.flags.append(flag.strip())
                elif child_tag in cls.SETTER_PARENT_TAGS:
                    self._update_setters(child_node)
                elif child_tag in cls.OTHER_ACCEPTED_TAGS:
                    if child_tag == "prerequisites":
                        _logger.warning("Node tag 'prerequisites' should be "
                                        "in singular form in %s", self.id)
                        child_tag = "prerequisite"
                    self.others[child_tag].append(child_node)
                else:
                    _logger.warning("Unexpected node tag '%s' in %s", child_tag, self.id)
        return self

    @classmethod
    def load_aurora_content(cls, *sub_dirs: str | Path, base: str | Path = "",
                            progress: ProgressTracker | None = None) -> dict[str, "PreParsing"]:
        xml_index = index_aurora_files(*sub_dirs, base=base, progress=progress)
        return cls.transform_index(xml_index, progress)


class AuroraPostParser(Generic[OutputT], AuroraParser[PreParsing, OutputT], metaclass=ABCMeta):
    @classmethod
    def _create_from_nodes(cls, nodes: XMLNodes) -> OutputT:
        return cls.create(PreParsing.create(nodes))

    @classmethod
    def load_aurora_content(cls, *sub_dirs: str | Path, base: str | Path = "",
                            progress: ProgressTracker | None = None) -> Mapping[str, OutputT]:
        xml_index = index_aurora_files(*sub_dirs, base=base, progress=progress)
        return transform_mapping_values(cls._create_from_nodes, xml_index, progress, "TRANSFORM")
    
    _OUTPUT_TYPE: type[OutputT]
    _DYNAMIC_MODELS: Iterable[type]

    @classmethod
    def load_cached_content(cls, cache_file: str | Path) -> Mapping[str, OutputT]:
        deserializer = Deserializer.from_subclasses(*cls._DYNAMIC_MODELS)
        return deserializer.json_load(cache_file, dict[str, cls._OUTPUT_TYPE])

    @classmethod
    def _write_content_cache(cls, content: Mapping[str, OutputT], cache_file: str) -> None:
        Serializer.json_dump(content, cache_file)

    @classmethod
    def load_aurora_content_with_caching(cls, *sub_dirs: Iterable[str | Path], base_dir: str | Path = "",
                                         cache_file: str | Path, progress: ProgressTracker | None = None) \
            -> Mapping[str, OutputT]:
        # 4-stage process: load, index, parse, cache
        try:
            return cls.load_cached_content(cache_file)  # Stage 1: (attempt to) load cache
        except Exception:
            _logger.warning("Couldn't load cache file due to the following exception. "
                            "Falling back to xml content.", exc_info=True)
        # Stage 2 + 3: index and parse (transform)
        content = cls.load_aurora_content(*sub_dirs, base=base_dir, progress=progress)
        cls._write_content_cache(content, cache_file)  # Stage 4. write cache
        return content


# SERIALIZING AND DESERIALIZING

class Serializer:
    @classmethod
    def serialize(cls, obj: object) -> object:
        if is_dataclass(obj):
            # Dataclass
            result = {f.name: cls.serialize(getattr(obj, f.name)) for f in fields(obj)
                      if "non-serializable" not in f.metadata}
            # Add concrete type as additional attribute
            result["__type__"] = obj.__class__.__name__
            return result
        elif isinstance(obj, Mapping):
            # Mapping instances (keys are converted to strings, values are serialized)
            return {str(k): cls.serialize(v) for k, v in obj.items()}
        elif isinstance(obj, str):
            # Because strings are sequences, but we want to keep them as they are
            return obj
        elif isinstance(obj, Sequence) or isinstance(obj, Set):
            # Sequences (including named tuples, excluding strings) and sets become tuples
            return tuple(cls.serialize(v) for v in obj)
        # int, bool, float, ... are returned unchanged
        return obj

    @classmethod
    def json_dumps(cls, obj: object, **kwargs) -> str:
        return json.dumps(cls.serialize(obj), **kwargs)

    @classmethod
    def json_dump(cls, obj: object, dump_file_path: str | Path, **kwargs) -> None:
        with open(dump_file_path, "w") as dump_file:
            json.dump(cls.serialize(obj), dump_file, **kwargs)


class Deserializer:
    def __init__(self, models: Iterable[type]):
        self.models = {cls.__name__: cls for cls in models}

    @classmethod
    def from_subclasses(cls, *classes: type) -> "Deserializer":
        return Deserializer(subclasses(*classes))

    def deserialize_dataclass(self, attrs: dict[str, object]) -> object:
        if "__type__" not in attrs:
            raise ValueError("attrs does not have a \"__type__\" key")
        model_name = attrs.pop("__type__")
        if model_name not in self.models:
            raise ValueError("Unknown model name {!r}".format(model_name))
        return self.deserialize(attrs, self.models[model_name])

    def deserialize(self, obj: object, model: type[OutputT]) -> OutputT:
        if isinstance(model, str):
            # Referenced type hint
            if model not in self.models:
                raise ValueError("Referenced type {!r} not in known models".format(model))
            model = self.models[model]
        # Mapping subtypes
        if isinstance(obj, Mapping):
            model = self._concrete_model(obj.get("__type__"), model.__name__) or model
            # !!! Note: handling generic dataclasses (mapping type var names to type args) seems impossible !!!
            # Dataclass instances
            if is_dataclass(model):
                parsed = {f.name: self.deserialize(obj[f.name], f.type) for f in fields(model)
                          if "non-serializable" not in f.metadata}
                return model(**parsed)
            # Generic mapping types
            elif isinstance(model, GenericAlias):
                key_model, value_model = model.__args__
                parsed = {self.deserialize(k, key_model): self.deserialize(v, value_model) for k, v in obj.items()}
                return model(parsed)
        # Sequence and set subtypes (sets are sequences if data comes from json)
        elif isinstance(obj, Sequence) or isinstance(obj, Set):
            # NamedTuple instances
            if issubclass(model, tuple) and hasattr(model, "_fields") and hasattr(model, "__annotations__"):
                annotations = model.__annotations__
                field_names = model._fields
                parsed = {f: self.deserialize(v, annotations[f]) for f, v in zip(field_names, obj)}
                return model(**parsed)
            # Generic sequence types
            elif isinstance(model, GenericAlias):
                type_args = model.__args__
                if type_args[-1] is Ellipsis:
                    type_args = type_args[:-1]
                type_args_cycle = itertools.cycle(type_args)
                args = (self.deserialize(attr, sub_type) for attr, sub_type in zip(obj, type_args_cycle))
                return model(args)
        # Note: isinstance(attrs, str) to deserialize mapping keys!
        # Other objects => bring to correct type
        return model(obj)

    def _concrete_model(self, class_name, fallback_name):
        if class_name and class_name != fallback_name:
            concrete_model = self.models.get(class_name)
            if concrete_model:
                return concrete_model
            _logger.warning("Concrete class {!r} not in known models, falling back to type-hinted class {!r}" \
                            .format(concrete_model, fallback_name))

    def json_loads(self, encoded: str, model: type[OutputT]) -> OutputT:
        obj = json.loads(encoded)
        return self.deserialize(obj, model)

    def json_load(self, load_file_path: str | Path, model: type[OutputT]) -> OutputT:
        with open(load_file_path) as load_file_handle:
            obj = json.load(load_file_handle)
        return self.deserialize(obj, model)


# MAIN - TESTING

if __name__ == '__main__':
    from pprint import pprint
    import timeit

    from auroratools.core.utils import configure_logging
    from auroratools.core.attrs_parsing import ItemAttrsParser, Item

    BASE = Path("../../../../content")
    PATHS = [
        BASE / Path("A-aurora-legacy/core"),
        BASE / Path("B-aurora-original/core"),
        BASE / Path("C-aurora-internal/core"),
    ]

    def test_aurora_index_sources():
        xml_index = index_aurora_files(*PATHS)
        pprint(list(xml_index.values())[:25], compact=True, width=240)
        print("#elements:", len(xml_index))

    def test_partial_parser():
        info_index = PreParsing.load_aurora_content(*PATHS)
        pprint(list(info_index.values())[:25], compact=True, width=240)
        print("#elements:", len(info_index))

    def test_de_serialization():
        deser = Deserializer.from_subclasses(Item)

        info_index = PreParsing.load_aurora_content(BASE)
        item_index = ItemAttrsParser.transform_index(info_index)
        dumped_index = json.dumps(Serializer.serialize(item_index))
        loaded_index = deser.deserialize(json.loads(dumped_index), dict[str, Item])

        for ident, item in list(loaded_index.items())[:25]:
            pprint(item, width=240)
        print("#elements:", len(loaded_index))

    def timeit_de_serialization():
        configure_logging(logging.ERROR)

        def load_and_parse():
            info_index = PreParsing.load_aurora_content(BASE)
            return ItemAttrsParser.transform_index(info_index)

        items_index = load_and_parse()

        def serialize_and_dump():
            return json.dumps(Serializer.serialize(items_index))

        dumped_index = serialize_and_dump()
        model_deserializer = Deserializer.from_subclasses(Item)

        def load_and_deserialize():
            return model_deserializer.deserialize(json.loads(dumped_index), dict[str, Item])

        number = 2
        parsing = timeit.timeit(load_and_parse, number=number // 2) / (number // 2)
        print("Loading and Parsing: {:7.2f}".format(parsing))
        serializing = timeit.timeit(serialize_and_dump, number=number) / number
        print("Serializing and Dumping: {:7.2f}".format(serializing))
        deserializing = timeit.timeit(load_and_deserialize, number=number) / number
        print("Loading and Deserializing: {:7.2f}".format(deserializing))

    test_aurora_index_sources()
    test_partial_parser()
    test_de_serialization()
    timeit_de_serialization()
