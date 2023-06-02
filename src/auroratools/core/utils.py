from collections.abc import MutableSequence as MutSeq, Mapping, Collection, Iterable
from typing import TypeVar, Callable, Any

from collections import defaultdict
import enum
import logging
import math


KeyT = TypeVar("KeyT")
DataT = TypeVar("DataT")
Key2T = TypeVar("Key2T")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

ProgressTracker = Callable[[Collection[DataT], str], Iterable[DataT]]


def track_progress_stdout(collection, title):
    count = len(collection)
    tenth = count / 10
    digits = math.ceil(math.log10(count)) if count else 0
    fmt = "{{:s}}: {{:{0:d}}} / {{:{0:d}}}".format(digits)
    for i, thing in enumerate(collection):
        if 0 <= i % tenth < 1:
            print(fmt.format(title, i, count))
        yield thing
    print(fmt.format(title, count, count))


def configure_logging(level: int | str, fmt: str = "%(levelname)s - %(name)s - %(message)s"):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt))
    console_handler.setLevel(level)
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(level)


def alter_sequence_shape(sequence: MutSeq, first_index: int, remove_len: int,
                         insert_len: int, fill: Any = None) -> None:
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

    def __new__(cls, *args, **kwargs):
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


def transform_mapping_values(transform: Callable[[InputT], OutputT], index: Mapping[KeyT, InputT],
                             progress: ProgressTracker | None = None, title: str = "") -> dict[KeyT, OutputT]:
    return {k: transform(v) for k, v in (progress(index.items(), title) if progress else index.items())}


def subclasses(*classes: type[DataT]) -> list[type[DataT]]:
    classes = list(classes)
    for cls in classes:
        classes.extend(cls.__subclasses__())
    return classes
