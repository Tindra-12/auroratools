from dataclasses import dataclass
from typing import Iterable

from auroratools.core.utils import grouped
from auroratools.core.base import Parser, PreParsing, AuroraPostParser
from auroratools.core.attrs_parsing import Item, ItemAttrsParser
from auroratools.core.nnf_querying import Query, QueryParser, QueryingEngine, TOP


class RequirementsParser(Parser[PreParsing, Query]):
    """Parses the requirements a content item has (producing a NNF Query)"""
    @classmethod
    def create(self, info: PreParsing) -> Query:
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
