from abc import ABCMeta, abstractmethod
from collections.abc import Iterator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import re
from typing import TypeVar, Generic, Union, Callable, Any

from auroratools.core.utils import singleton

__all__ = ["Query", "Logic", "Con", "Dis", "Top", "Bottom", "TOP", "BOTTOM", "LogicBuilder",
           "Lookup", "Ref", "Test", "Unknown", "QuerySyntaxError", "QueryParser", "QueryingEngine"]


# QUERY BASE CLASS AND LOGICAL QUERIES

ItemT = TypeVar("ItemT")
AssignmentGetter = Callable[[ItemT], Iterable[str]]


class Query(metaclass=ABCMeta):
    @abstractmethod
    def accepts(self, assignment: Iterable[str]) -> bool:
        raise NotImplementedError()

    def select(self, items: Iterable[ItemT], getter: AssignmentGetter) -> Iterable[ItemT]:
        return filter(lambda item: self.accepts(getter(item)), items)


@dataclass(frozen=True)
class Logic(Query, metaclass=ABCMeta):
    pos: frozenset[str] = field(default=frozenset())
    neg: frozenset[str] = field(default=frozenset())
    sub: tuple[Query, ...] = field(default=tuple())
    sep = " "

    def __len__(self) -> int:
        return len(self.pos) + len(self.neg) + len(self.sub)

    def __str__(self) -> str:
        a = self.pos, ("!{}".format(v) for v in self.neg), (str(s) for s in self.sub)
        return "{}[{}]".format(self.__class__.__name__.lower(), self.sep.join(c for b in a for c in b))


@dataclass(frozen=True)
class Con(Logic):
    sep = ","

    def accepts(self, assignment: Iterable[str]) -> bool:
        return self.pos.issubset(assignment) and self.neg.isdisjoint(assignment) \
               and all(sub.accepts(assignment) for sub in self.sub)


@dataclass(frozen=True)
class Dis(Logic):
    sep = "|"

    def accepts(self, assignment: Iterable[str]) -> bool:
        return not self.pos.isdisjoint(assignment) or not self.neg.issubset(assignment) \
               or any(sub.accepts(assignment) for sub in self.sub)


@singleton
@dataclass(frozen=True)
class Top(Query):
    """A query that accepts everything"""

    def accepts(self, assignment: Iterable[str]) -> bool:
        return True

    def select(self, items: Iterable[ItemT], getter: AssignmentGetter) -> Iterable[ItemT]:
        return items


@singleton
@dataclass(frozen=True)
class Bottom(Query):
    """A query that denies anything, i.e. accepts nothing"""

    def accepts(self, assignment: Iterable[str]) -> bool:
        return False

    def select(self, items: Iterable[ItemT], getter: AssignmentGetter) -> Iterable[ItemT]:
        return ()


# Singleton variables
TOP = Top()
BOTTOM = Bottom()


class LogicBuilder:
    def __init__(self, cls: type[Logic], pos: Iterable[str] = (), neg: Iterable[str] = (),
                 sub: Iterable[Union[Query, "LogicBuilder"]] = ()):
        self.cls = cls
        self.pos = set[str]()
        self.pos.update(pos)
        self.neg = set[str]()
        self.neg.update(neg)
        self.sub = list[Query]()
        for sub in sub:
            self.add_sub(sub)

    def __len__(self) -> int:
        return len(self.pos) + len(self.neg) + len(self.sub)

    def add_sub(self, sub: Union[Query, "LogicBuilder"]):
        if not (isinstance(sub, Logic) or isinstance(sub, LogicBuilder)) or len(sub) > 1:
            if isinstance(sub, LogicBuilder):
                sub = sub._build()
            self.sub.append(sub)
            return
        if len(sub.sub) == 1:
            sub_sub = sub.sub[0]
            if isinstance(sub_sub, Logic):
                sub = sub_sub
        self.pos.update(sub.pos)
        self.neg.update(sub.neg)
        self.sub.extend(sub.sub)

    def build(self) -> Logic:
        return self.sub.pop() if len(self) == 1 and self.sub else self._build()

    def _build(self) -> Logic:
        return self.cls(frozenset(self.pos), frozenset(self.neg), tuple(self.sub))

    def clear(self):
        self.pos.clear()
        self.neg.clear()
        self.sub.clear()

    # Factory class methods

    @classmethod
    def conjunctive(cls) -> "LogicBuilder":
        return cls(Con)

    @classmethod
    def disjunctive(cls) -> "LogicBuilder":
        return cls(Dis)


# LOOKUP QUERY CLASSES

@dataclass(frozen=True)
class Lookup(Query, metaclass=ABCMeta):
    name: str
    true: bool

    @staticmethod
    def lookup(name: str) -> Any:
        """Set lookup on a subclass or an instance"""
        raise KeyError(name)

    @classmethod
    @contextmanager
    def lookup_context(cls, lookup: Callable[[str], Any]):
        if cls is Lookup:
            raise TypeError("Don't call this method on Lookup but on its subclasses!")
        cls.lookup = lookup
        try:
            yield None
        finally:
            del cls.lookup


@dataclass(frozen=True)
class Ref(Lookup):
    """Query forwarding / Reference to another query"""

    def accepts(self, assignment: Iterable[str]) -> bool:
        return self.lookup(self.name).accepts(assignment) == self.true


@dataclass(frozen=True)
class Test(Lookup):
    threshold: int

    def accepts(self, assignment: Iterable[str]) -> bool:
        return (self.lookup(self.name) >= self.threshold) == self.true


@dataclass(frozen=True)
class Unknown(Lookup):
    """TODO handle these test-brackets: [armor:like] and [type:spell] query"""
    value: str

    def accepts(self, assignment: Iterable[str]) -> bool:
        raise NotImplementedError()


# QUERY PARSING

class QuerySyntaxError(RuntimeError):
    def __init__(self, token: str, expected: str):
        super().__init__("Expected {} but got {}".format(expected, repr(token) if token else "EOF"))


class QueryParser:
    SPECIAL_TOKENS = re.compile(r"(\|\|?|&&?|[,!()\[\]])")
    TOKENS_MAP = {"||": "|", "&&": ",", "&": ","}
    EXPRESSION_MSG = "literal or '(' or '['"
    SEPARATOR_MSG = " separator ',' or '|' or "  # ) or EOF

    @classmethod
    def scan_and_parse(cls, query: str, default: Query = TOP) -> Query:
        return cls.scan_and_parse_nonempty(query) if query else default

    @classmethod
    def scan_and_parse_nonempty(cls, query: str) -> Query:
        return cls.syntax_analysis(cls.tokenize(query))

    @classmethod
    def scan_and_build(cls, query: str) -> LogicBuilder:
        return cls.scan_and_build_nonempty(query) if query else LogicBuilder(Con)

    @classmethod
    def scan_and_build_nonempty(cls, query: str) -> LogicBuilder:
        return cls.syntax_analysis_builder(cls.tokenize(query))

    @classmethod
    def syntax_analysis(cls, tokens: Iterator[str]) -> Query:
        return cls.syntax_analysis_builder(tokens).build()

    @classmethod
    def syntax_analysis_builder(cls, tokens: Iterator[str], positive: bool = True,
                                outermost: bool = True) -> LogicBuilder:
        outer_set = LogicBuilder(Dis)
        inner_set = LogicBuilder(Con)
        while True:
            # 1. Token: keywords or parentheses expected
            try:
                token_positive = positive
                token = next(tokens)
                if token == "!":
                    token_positive = not positive
                    token = next(tokens)
                if token == "(":
                    inner_set.add_sub(cls.syntax_analysis_builder(tokens, token_positive, False))
                elif token == "$":
                    cls._check_token(tokens, "(", "to introduce placeholder name")
                    inner_set.add_sub(Ref(next(tokens), token_positive))
                    cls._check_token(tokens, ")", "to conclude placeholder name")
                elif token == "[":
                    inner_set.add_sub(cls._create_test_or_unknown(next(tokens), token_positive))
                    cls._check_token(tokens, "]", "to conclude test expression")
                elif token not in (",", "|", ")", "!"):
                    if token_positive:
                        inner_set.pos.add(token)
                    else:
                        inner_set.neg.add(token)
                else:
                    raise QuerySyntaxError(token, cls.EXPRESSION_MSG)
            except StopIteration:
                raise QuerySyntaxError("", cls.EXPRESSION_MSG)
            # 2. Separator: , or || or ) or EOF
            try:
                token = next(tokens)
                if (positive and token == ",") or (not positive and token == "|"):
                    continue
                elif (positive and token == "|") or (not positive and token == ","):
                    outer_set.add_sub(inner_set)
                    inner_set.clear()
                elif token == ")":
                    if not outermost:
                        break
                    raise QuerySyntaxError(token, "any other" + cls.SEPARATOR_MSG + "EOF")
                else:
                    raise QuerySyntaxError(token, "a" + cls.SEPARATOR_MSG + "')' or EOF")
            except StopIteration:
                if outermost:
                    break
                raise QuerySyntaxError("", "any other" + cls.SEPARATOR_MSG + "')'")
        outer_set.add_sub(inner_set)
        return outer_set

    @classmethod
    def _check_token(cls, tokens: Iterator[str], expected: str, message: str):
        try:
            token = next(tokens)
            if token != expected:
                raise QuerySyntaxError(token, repr(expected) + " " + message)
        except StopIteration:
            raise QuerySyntaxError("", repr(expected) + " " + message)

    @classmethod
    def _create_test_or_unknown(cls, token: str, true: bool) -> Lookup:
        name, value = token.rsplit(":", 1)
        try:
            return Test(name, true, int(value))
        except ValueError:
            return Unknown(name, true, value)

    @classmethod
    def tokenize(cls, query: str) -> Iterator[str]:
        return filter(None, map(cls._preprocess_token, cls.SPECIAL_TOKENS.split(query)))

    @classmethod
    def _preprocess_token(cls, token: str) -> str:
        token = token.strip()
        return cls.TOKENS_MAP.get(token, token)


# Item Querying from database

KeyT = TypeVar("KeyT")


class QueryingEngine(Generic[KeyT, ItemT], metaclass=ABCMeta):
    def __init__(self, grouped_items: Mapping[KeyT, Iterable[ItemT]]):
        self.database = grouped_items

    def select(self, group: KeyT, query: Query = TOP) -> Iterable[ItemT]:
        return query.select(self.database.get(group, ()), self.get_flags)

    def scan_parse_select(self, group: KeyT, query: str = "") -> Iterable[ItemT]:
        items = self.database.get(group, ())
        return QueryParser.scan_and_parse_nonempty(query).select(items, self.get_flags) if query else items

    @abstractmethod
    def get_flags(self, item: ItemT) -> Iterable[str]:
        raise NotImplementedError()


# TESTING

if __name__ == '__main__':
    from collections import defaultdict
    from pprint import pprint

    from auroratools.core.utils import index_aurora_files

    class ConstructorHook:
        def __init__(self, cls: type):
            self.cls = cls
            self.instances = {}
            self.original_new = self.cls.__new__

        def hooked_new(self, cls, *args):
            if args not in self.instances:
                self.instances[args] = object.__new__(cls)
            return self.instances[args]

        def enable(self):
            self.cls.__new__ = self.hooked_new

        def disable(self):
            self.cls.__new__ = self.original_new

    class ItemQueryTester:
        def __init__(self):
            self.parser = QueryParser()
            self.queries = defaultdict[Query, list[str]](list)
            self.missing_count = 0
            self.error_count = 0
            self.hooks = {
                "# $(reference)": ConstructorHook(Ref),
                "# [test:value]": ConstructorHook(Test),
                "# [Unknowns]  ": ConstructorHook(Unknown)
            }

        def add_query(self, text: str | None, item_id: str):
            # Count missing or empty queries
            if text is None:
                self.missing_count += 1
                return
            # Enable query construction hooks
            for hook in self.hooks.values():
                hook.enable()
            try:
                # Parse and construct queries
                parsed = self.parser.scan_and_parse(text)
                self.queries[parsed].append(text)
            except QuerySyntaxError as qse:
                print("QSE '{}' in '{}'".format(qse, item_id))
                self.error_count += 1
            # Disable query construction hooks
            for hook in self.hooks.values():
                hook.disable()

        def print_results(self, name: str):
            counts = {
                "error": self.error_count,
                "missing": self.missing_count,
                "successful": sum(map(len, self.queries.values()))
            }
            total = sum(counts.values())
            counts["total"] = total
            print("\n" + name)
            for title, count in counts.items():
                print("{:10s}: {:5d}, {:6.2f}%".format(title, count, 100 * count / total))
            print("distinct successful queries", len(self.queries))

            count_per_depth = defaultdict[int, int](int)
            for query, variants in self.queries.items():
                count_per_depth[self.query_depth(query)] += 1
            print("distinct query count per depth:",
                  {k: count_per_depth[k] for k in sorted(count_per_depth)})

            for name, hook in self.hooks.items():
                print("{}: {}".format(name, len(hook.instances)))
                if hook.instances:
                    if name.startswith("# [Unknowns]"):
                        pprint(set(hook.instances), width=180)
                    else:  # Cut off __getitem__ method argument
                        pprint(set(inst[:-1] for inst in hook.instances), width=180)

        @staticmethod
        def query_depth(query: Query) -> int:
            return (1 + max(map(ItemQueryTester.query_depth, query.sub))) \
                if isinstance(query, Logic) and query.sub else 1  # 0.5 if isinstance(query, Ref) else 1

    def test_aurora_queries():
        base = "../../../../content"
        index = index_aurora_files("A-aurora-legacy", "C-aurora-internal", base)

        item_reqs = ItemQueryTester()
        rule_reqs = ItemQueryTester()
        equipped_qs = ItemQueryTester()
        select_qs = ItemQueryTester()

        for item_id, elements in index.items():
            for element_node in elements:
                if element_node.tag not in ("element", "append"):
                    continue
                # Item requirements
                requires_node = element_node.find("requirements")
                item_reqs.add_query(requires_node.text if requires_node is not None else None, item_id)
                for node_query in ("rules/grant", "rules/select", "rules/stat"):
                    for rule_node in element_node.iterfind(node_query):
                        # Rule requirements
                        rule_reqs.add_query(rule_node.attrib.get("requirements"), item_id)
                        # Equipped queries
                        equipped_qs.add_query(rule_node.attrib.get("equipped"), item_id)
                for select_node in element_node.findall("rules/select"):
                    # Rule select queries
                    select_qs.add_query(select_node.attrib.get("supports"), item_id)

        item_reqs.print_results("ITEM REQUIREMENTS")
        rule_reqs.print_results("RULE REQUIREMENTS")
        select_qs.print_results("SELECT QUERIES")
        equipped_qs.print_results("EQUIPPED QUERIES")

        # Notes:
        # - Item and Rule Requirements have [test]s but NO $(reference)s
        # - Equipped queries (whatever they are), too, have [test]s but NO $(reference)s
        # - Select queries have NO [test]s but the following two $(reference)s
        #   'spellcasting:list', 'spellcasting:slots' (both positively only)
        #   (Presumably, they are only relevant for type='Spell' choices)

    test_aurora_queries()
