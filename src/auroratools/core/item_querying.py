from dataclasses import dataclass
from collections.abc import Iterable

from auroratools.core.utils import grouped
from auroratools.core.base import Parser, PreParsing, AuroraPostParser
from auroratools.core.attrs_parsing import Item, ItemAttrsParser
from auroratools.core.nnf_querying import Query, QueryParser, QueryingEngine, TOP


class RequirementsParser(Parser[PreParsing, Query]):
    """Parses the requirements a content item has (producing a NNF Query)"""
    @classmethod
    def create(cls, info: PreParsing) -> Query:
        text_requirements = []
        for requirement_node in info.others["requirements"]:
            text_requirements.append(requirement_node.text or "")
        return QueryParser.scan_and_parse(",".join(text_requirements))


class FlagsParser(Parser[PreParsing, frozenset[str]]):
    """Parses the flags a content item supports (with special case handling for spells)"""
    @classmethod
    def create(cls, info: PreParsing) -> frozenset[str]:
        flags = list(info.flags)
        # Some stupid special-case handling for spells
        #  (necessary for choice option selection)
        # TODO are id and name really only for spells?
        if info.kind == "Spell":
            flags.append(info.id)
            # flags.append(info.name)
            flags.append(info.attrs["level"])
            flags.append(info.attrs["school"])
        return frozenset(flags)


@dataclass(frozen=True)
class ItemQueryingData(AuroraPostParser["ItemQueryingData"]):
    """This class represents all the data of a content item that is needed to query items"""
    id: str
    kind: str
    flags: frozenset[str]
    requs: Query  # requirements
    attrs: Item

    _DYNAMIC_MODELS = Item, Query

    @classmethod
    def create(cls, info: PreParsing) -> "ItemQueryingData":
        return cls(info.id, info.kind, FlagsParser.create(info),
                   RequirementsParser.create(info), ItemAttrsParser.create(info))

    def get_id(self) -> str:
        return self.id

    def get_kind(self) -> str:
        return self.kind

    def get_flags(self) -> frozenset[str]:
        return self.flags

    def get_attrs(self) -> Item:
        return self.attrs


ItemQueryingData._OUTPUT_TYPE = ItemQueryingData


class ItemQueryingEngine(QueryingEngine[str, ItemQueryingData]):
    def __init__(self, items: Iterable[ItemQueryingData]):
        super().__init__(grouped(items, ItemQueryingData.get_kind), ItemQueryingData.get_flags)

    def select_attrs(self, group: str, query: Query = TOP) -> Iterable[Item]:
        return map(ItemQueryingData.get_attrs, self.select(group, query))

    def scan_parse_select_attrs(self, group: str, query: str = "") -> Iterable[Item]:
        return map(ItemQueryingData.get_attrs, self.scan_parse_select(group, query))


if __name__ == '__main__':
    import logging
    from pathlib import Path

    from auroratools.core.utils import configure_logging, track_progress_stdout

    configure_logging(logging.ERROR)

    BASE = Path("../../../")

    def test_load_with_cache():
        # Clean up cache from last usage
        cache_file_path = BASE / "results/small_items_cache.json"
        print(cache_file_path.absolute())
        cache_file_path.unlink(missing_ok=True)

        # Call for the first time
        a = ItemQueryingData.load_aurora_content_with_caching(
            BASE / "../content/A-aurora-legacy/core", cache_file=cache_file_path, progress=track_progress_stdout)
        print("Result 1: #", len(a))

        # Call for the second time
        a = ItemQueryingData.load_aurora_content_with_caching(
            BASE / "../content/A-aurora-legacy/core", cache_file=cache_file_path, progress=track_progress_stdout)
        print("Result 2: #", len(a))

    test_load_with_cache()
