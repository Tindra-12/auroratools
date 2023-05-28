from collections.abc import Mapping, Sequence, MutableSequence, Set, Collection, Iterable
from dataclasses import dataclass, field
from typing import TypeVar, Generic, overload, Callable
from types import GenericAlias

from collections import defaultdict
import dataclasses
import enum
import itertools
import json
import logging
from pathlib import Path
from xml.etree.ElementTree import ElementTree, Element

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

__all__ = ["ProgressTracker", "transform_mapping_values", "subclasses", "alter_sequence_shape",
           "grouped", "grouped_mapping", "AutoIntEnum", "singleton", "configure_logging",
           "Parser", "StaticParser", "index_aurora_files", "PreParsing", "Serializer", "Deserializer"]


KeyT = TypeVar("KeyT")
DataT = TypeVar("DataT")

Key2T = TypeVar("Key2T")
Data2T = TypeVar("Data2T")

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

ProgressTracker = Callable[[Collection[DataT]], Iterable[DataT]]


def configure_logging(level: int | str, fmt: str = "%(levelname)s - %(name)s - %(message)s"):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt))
    console_handler.setLevel(level)
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(level)


def alter_sequence_shape(sequence: MutableSequence[DataT], first_index: int,
                         remove_len: int, insert_len: int, fill: Data2T = None) -> None:
    # Removal part
    del sequence[first_index:first_index + remove_len]
    # Insertion part
    for i in range(first_index, first_index + insert_len):
        sequence.insert(i, fill)


class AutoIntEnum(int, enum.Enum):
    """
    An int enum that automatically numbers its members in the order declared,
    starting at 0. It was inspired by aenum (https://github.com/ethanfurman/aenum).
    """
    def __new__(cls, *args):
        ordinal = len(cls.__members__)
        obj = super().__new__(cls, ordinal)
        obj._value_ = ordinal
        return obj


def singleton(cls):
    cls._instance = cls()

    def __new__(cls):
        return cls._instance

    __new__.__qualname__ = cls.__name__ + "." + __new__.__name__
    cls.__new__ = __new__

    return cls


def grouped(items: Iterable[DataT], key: Callable[[DataT], KeyT]) -> defaultdict[KeyT, list[DataT]]:
    sorted_items: defaultdict[KeyT, list[DataT]] = defaultdict(list)
    for item in items:
        sorted_items[key(item)].append(item)
    return sorted_items


def grouped_mapping(items: Iterable[DataT], key_1: Callable[[DataT], KeyT],
                    key_2: Callable[[DataT], Key2T]) -> dict[KeyT, dict[Key2T, DataT]]:
    sorted_items: dict[KeyT, dict[Key2T, DataT]] = defaultdict(dict)
    for item in items:
        sorted_items[key_1(item)][key_2(item)] = item  # We assume no duplicated item key_2s here!
    return {**sorted_items}


# PARSER INTERFACES

def transform_mapping_values(index: Mapping[KeyT, InputT], transform: Callable[[InputT], OutputT],
                             progress: ProgressTracker | None = None) -> dict[KeyT, OutputT]:
    return {k: transform(v) for k, v in (progress(index.items()) if progress else index.items())}


def subclasses(*classes: type[DataT]) -> list[type[DataT]]:
    classes = list(classes)
    for cls in classes:
        classes.extend(cls.__subclasses__())
    return classes


class Parser(Generic[InputT, OutputT]):
    def create(self, something: InputT) -> OutputT:
        raise NotImplementedError()

    def transform_index(self, input_index: Mapping[KeyT, InputT],
                        progress: ProgressTracker | None = None) -> dict[KeyT, OutputT]:
        return transform_mapping_values(input_index, self.create, progress)


class StaticParser(Generic[InputT, OutputT], Parser[InputT, OutputT]):
    @classmethod
    def create(cls, something: InputT) -> OutputT:
        raise NotImplementedError()

    @classmethod
    def transform_index(cls, input_index: Mapping[KeyT, InputT],
                        progress: ProgressTracker | None = None) -> dict[KeyT, OutputT]:
        return transform_mapping_values(input_index, cls.create, progress)


# CONTENT FILE INDEXING AND PRE-PARSING

XMLIndex = dict[str, Sequence[Element]]


def _fuse_index(main_nodes: Mapping[str, Element], appendixes: Mapping[str, Collection[Element]]) -> XMLIndex:
    return {aurora_id: (main_node, *appendixes.get(aurora_id, ())) for aurora_id, main_node in main_nodes.items()}


def index_aurora_files(*content_dirs: str | Path, base: str | Path = "",
                       progress: ProgressTracker | None = None) -> XMLIndex:
    base = Path(base)
    files = sorted(file for path in content_dirs for file in (base / Path(path)).glob("**/*.xml"))
    main_nodes = dict[str, Element]()
    appendixes = defaultdict[str, list[Element]](list)
    for file in (progress(files) if progress else files):
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
class PreParsing(StaticParser[Sequence[Element], "PreParsing"]):
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
    def create(cls, nodes: Sequence[Element]) -> "PreParsing":
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
    def index_aurora_files(cls, *content_dirs: str | Path, base: str | Path = "",
                           progress: ProgressTracker | None = None) -> dict[str, "PreParsing"]:
        return cls.transform_index(index_aurora_files(*content_dirs, base=base, progress=progress), progress)


# SERIALIZING AND DESERIALIZING

class Serializer(StaticParser[object, dict[str, object]]):
    @classmethod
    def create(cls, obj: object) -> object:
        return cls.serialize(obj)

    @classmethod
    def serialize(cls, obj: object) -> object:
        if dataclasses.is_dataclass(obj):
            # Dataclass
            result = {f.name: cls.serialize(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
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
        with open(dump_file_path, "w") as dump_file_handle:
            json.dump(cls.serialize(obj), dump_file_handle, **kwargs)


ObjectT = TypeVar("ObjectT", bound=object)
ObjectV = TypeVar("ObjectV", bound=object)


class Deserializer(Generic[ObjectT], Parser[dict[str, object], ObjectT]):
    def __init__(self, models: Iterable[type[ObjectT]]):
        self.models = {cls.__name__: cls for cls in models}

    @classmethod
    def with_subclasses_of(cls, *classes: type[ObjectT]) -> "Deserializer[ObjectT]":
        return Deserializer(subclasses(*classes))

    def create(self, attrs: dict[str, object]) -> ObjectT:
        if "__type__" not in attrs:
            raise ValueError("given dict does not have a __type__ key")
        model_name = attrs.pop("__type__")
        if model_name not in self.models:
            raise ValueError("Concrete type {!r} not in known models".format(model_name))
        return self.deserialize(attrs, self.models[model_name])

    def deserialize(self, obj: object, model: type[ObjectV]) -> ObjectV:
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
            if dataclasses.is_dataclass(model):
                parsed = {f.name: self.deserialize(obj[f.name], f.type)
                          for f in dataclasses.fields(model) if f.metadata.get("serialize", True)}
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

    @overload
    def json_loads(self, encoded: str) -> ObjectT: ...

    @overload
    def json_loads(self, encoded: str, model: type[ObjectV]) -> ObjectV: ...

    def json_loads(self, encoded: str, model: type[ObjectV] | None = None) -> ObjectV | ObjectT:
        obj = json.loads(encoded)
        return self.create(obj) if model is None else self.deserialize(obj, model)

    @overload
    def json_load(self, load_file_path: str | Path) -> ObjectT: ...

    @overload
    def json_load(self, load_file_path: str | Path, model: type[ObjectV]) -> ObjectV: ...

    def json_load(self, load_file_path: str | Path, model: type[ObjectV] | None = None) -> ObjectV | ObjectT:
        with open(load_file_path) as load_file_handle:
            obj = json.load(load_file_handle)
        return self.create(obj) if model is None else self.deserialize(obj, model)


# MAIN - TESTING

if __name__ == '__main__':
    from pprint import pprint
    import timeit

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
        info_index = PreParsing.index_aurora_files(*PATHS)
        pprint(list(info_index.values())[:25], compact=True, width=240)
        print("#elements:", len(info_index))

    def test_de_serialization():
        model_deserializer = Deserializer.with_subclasses_of(Item)

        info_index = PreParsing.index_aurora_files(BASE)
        item_index = ItemAttrsParser.transform_index(info_index)
        dumped_index = json.dumps(Serializer.transform_index(item_index))
        loaded_index = model_deserializer.transform_index(json.loads(dumped_index))

        for ident, item in list(loaded_index.items())[:25]:
            pprint(item, width=240)
        print("#elements:", len(loaded_index))

    def timeit_de_serialization():
        configure_logging(logging.ERROR)

        def load_and_parse():
            info_index = PreParsing.index_aurora_files(BASE)
            return ItemAttrsParser.transform_index(info_index)

        items_index = load_and_parse()

        def serialize_and_dump():
            return json.dumps(Serializer.transform_index(items_index))

        dumped_index = serialize_and_dump()
        model_deserializer = Deserializer.with_subclasses_of(Item)

        def load_and_deserialize():
            return model_deserializer.transform_index(json.loads(dumped_index))

        number = 200
        parsing = timeit.timeit(load_and_parse, number=number // 2) / (number // 2)
        print("Loading and Parsing: {:7.2f}".format(parsing))
        serializing = timeit.timeit(serialize_and_dump, number=number) / number
        print("Serializing and Dumping: {:7.2f}".format(serializing))
        deserializing = timeit.timeit(load_and_deserialize, number=number) / number
        print("Loading and Deserializing: {:7.2f}".format(deserializing))

    # test_aurora_index_sources()
    # test_partial_parser()
    # test_de_serialization()
    timeit_de_serialization()
