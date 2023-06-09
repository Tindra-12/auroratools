from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Sequence, Mapping
from dataclasses import dataclass, field
from typing import NamedTuple, TypeVar, Generic, Callable

from collections import defaultdict
from pathlib import Path
from xml.etree.ElementTree import Element
from xml.etree import ElementTree as ETreeModule
import enum
import logging
import re

from auroratools.core.base import AuroraPostParser, PreParsing, Deserializer

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


# HELPER TYPES

class Dice(NamedTuple):
    number: int = 0
    sides: int = 1

    @classmethod
    def parse(cls, text: str) -> "Dice":
        return Dice(*map(int, text.split("d"))) if text else Dice()

    def format(self) -> str:
        if not self.number:
            return ""
        elif self.sides == 1:
            return str(self.number)
        return "{}d{}".format(self.number, self.sides)

    def average(self):
        return self.number * (self.sides + 1) / 2

    def __str__(self):
        return self.format() or "—"

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.format())

    def __bool__(self):
        return bool(self.number)


class Quantity(NamedTuple):
    value: float = 0.0
    unit: str = ""

    @classmethod
    def parse_number(cls, value: str, unit: str = "") -> "Quantity":
        return cls(float(value.replace(",", "")) if value != "one" else 1.0, unit)

    @classmethod
    def parse_regex_match(cls, match: re.Match | None) -> "Quantity":
        return cls.parse_number(*match.groups()) if match else cls()

    @classmethod
    def text_only(cls, text: str) -> "Quantity":
        return cls(0.0, text)

    def unitless(self, weights: Mapping[str, float | int]) -> float:
        return self.value * weights.get(self.unit, 0)

    def format(self, template: str = "{:g} {}", value_template: str = "{:g}") -> str:
        if self.value and self.unit:
            return template.format(self.value, self.unit)  # Normal
        elif self.value:
            return value_template.format(self.value)  # Plain number
        elif self.unit:
            return self.unit  # Text only
        return ""  # Empty

    def standard_format(self, template: str = "{:g} {}"):
        return template.format(self.value, self.unit)  # Normal

    def __str__(self):
        return self.format() or "—"

    def __repr__(self):
        return "{}({!r}, {!r})".format(self.__class__.__name__, self.value, self.unit)

    def __bool__(self):
        return bool(self.value or self.unit)


# ITEM DATA MODEL CLASSES

@dataclass(frozen=True)
class Item:
    id: str
    # kind: str
    # flags: frozenset[str]
    name: str
    source: str
    category: str
    text: tuple[str] = field(repr=False)


@dataclass(frozen=True)
class Equipment(Item):
    cost: Quantity
    weight: float  # unit: lb


@dataclass(frozen=True)
class AdventuringGear(Equipment):
    label = "Adventuring Gear"
    quantity: Quantity


@dataclass(frozen=True)
class EquipmentPack(Equipment):
    label = "Equipment Packs"

    @dataclass(frozen=True)
    class Content:
        reference: str
        quantity: Quantity

    contents: tuple[Content, ...]


@dataclass(frozen=True)
class Poison(Equipment):
    pass


@dataclass(frozen=True)
class MountOrVehicle(Equipment):
    label = "Mounts and Vehicles"

    speed: Quantity
    capacity: int  # carrying capacity | unit: lb | optional, default: 0


@dataclass(frozen=True)
class ExpertEquipment(Equipment):
    proficiency: str


@dataclass(frozen=True)
class Tool(ExpertEquipment):
    pass


@dataclass(frozen=True)
class Armor(ExpertEquipment):
    # categories = "Shield", "Light", "Medium", "Heavy"
    ac: int
    min_strength: int
    stealth_disadvantage: bool


@dataclass(frozen=True)
class Weapon(ExpertEquipment):
    # categories = "Simple Melee", "Simple Ranged", "Martial Melee", "Martial Ranged", "Firearm"
    # categories = enum.Enum("WeaponCategory", categories)
    # categories.__str__ = lambda self: self._name_

    class Damage(NamedTuple):
        type: str
        normal: Dice
        versatile: Dice

        def __bool__(self) -> bool:
            return bool(self.type)

    class Range(NamedTuple):
        normal: int
        long: int

        def __bool__(self) -> bool:
            return bool(self.normal or self.long)

    # category = simple / martial + melee / ranged or firearm
    # PROPERTIES: General: Finesse, Heavy, Light, Loading, Reach, Two-Handed, Special, Versatile
    # Ranged: Ammunition, Thrown. Firearm: Burst Fire, Reload. Gunslinger: Explosive, Scatter
    properties: tuple[str, ...]
    links: tuple[str, ...]  # The item ids to the proper weapon property items
    damage: Damage
    range: Range

    @property
    def melee(self) -> bool:
        return self.category.startswith("Simple")


@dataclass(frozen=True)
class Gun(Weapon):
    misfire: int
    reload: int


@dataclass(frozen=True)
class MagicItem(Item):
    # rarities = "common", "uncommon", "rare", "very rare", "legendary", "artifact", "unique", \
    #            "artificer infusion", "varying", ""  # everything that is not in this list is 'varying'
    # category according to the SRD
    spec: str  # Additional specification of the category
    rarity: str
    attune: str  # by whom attunement is possible or empty if not required


@dataclass(frozen=True)
class Spell(Item):
    # category = school
    level: int
    concentration: bool
    ritual: bool
    verbal: bool
    somatic: bool
    # material: QuantitySpec[str]  # quan: estimated material costs
    material: str
    cost: Quantity  # estimated material cost
    time: Quantity
    when: str  # when to take in case of reactions (or some other time specs)
    range: Quantity
    aoe: Quantity  # area of effect
    # duration: QuantitySpec[bool]  # Spec: True for up to / until lasting effects
    duration: Quantity
    duration_flag: bool  # True for up to / until lasting effects

    @property
    def school(self) -> str:
        return self.category


@dataclass(frozen=True)
class Language(Item):
    categories = "Standard", "Exotic", "Secret", "Monster", ""
    script: str
    speakers: str


@dataclass(frozen=True)
class PlayerClass(Item):
    hit_die: int  # number of sides
    short: str


@dataclass(frozen=True)
class Level(Item):
    level: int
    experience: int


@dataclass(frozen=True)
class Feature(Item):
    title: str
    usage: str
    action: str
    shorts: tuple[tuple[int, str], ...]  # Mapping[int, str]  # TODO getting the short for a in-between level


@dataclass(frozen=True)
class Proficiency(Item):
    pass


@dataclass(frozen=True)
class Skill(Proficiency):
    stat: str


@dataclass(frozen=True)
class FakeEquipment(Item):
    pass


@dataclass(frozen=True)
class WeaponProperty(Item):
    value: Quantity


@dataclass(frozen=True)
class Condition(Item):
    pass


# PREDEFINED ITEM NAMES / GROUPS / HIERARCHY

ITEM_GROUPS = {
    "Class": PlayerClass,
    "Condition": Condition,
    "Equipment": Equipment,
    " - Adventuring Gear": AdventuringGear,
    " - Armor": Armor,
    " - Equipment Packs": EquipmentPack,
    " - Mounts and Vehicles": MountOrVehicle,
    " - Poison": Poison,
    " - Tools": Tool,
    " - Weapons": Weapon,
    "    - Guns": Gun,
    "Fake Equipment": FakeEquipment,
    "Languages": Language,
    "Level": Level,
    "Magic Items": MagicItem,
    "Proficiencies": Proficiency,
    " - Skills": Skill,
    "Spells": Spell,
    "Weapon Property": WeaponProperty,
}


_NAME_TO_CLASS_MAP: Mapping[str, type[Item]] = {
    "Equipment": Equipment,
    "Armor": Armor,
    "Weapons": Weapon,
    "Tools": Tool,
    "Mounts and Vehicles": MountOrVehicle,
    "Adventuring Gear": AdventuringGear,
    "Equipment Packs": EquipmentPack,
    "Poisons": Poison,
    "Magic Items": MagicItem,
    "Spells": Spell,
    "Languages": Language,
    "Proficiencies": Proficiency,
    # Class, Level, Weapon Property have a sole category
    # Conditions by category => Immunity, Resistance, Vulnerability
    # FakeEquipment by category
    # Item, Feature and everything else by category, too
}


_SING_NAME_TO_CLASS_MAP: Mapping[str, type[Item]] = {
    "Equipment": Equipment,
    "Armor": Armor,
    "Weapon": Weapon,
    "Tool": Tool,
    "Mount Or Vehicle": MountOrVehicle,
    "Adventuring Gear": AdventuringGear,
    "Equipment Pack": EquipmentPack,
    "Poison": Poison,
    "Magic Item": MagicItem,
    "Spell": Spell,
    "Language": Language,
    "Proficiency": Proficiency,
    # Class, Level, Weapon Property have a sole category
    # Conditions by category => Immunity, Resistance, Vulnerability
    # FakeEquipment by category
    # Item, Feature and everything else by category, too
}


_CLASS_TO_NAME_MAP: Mapping[type[Item], str] = {cls: name for name, cls in _NAME_TO_CLASS_MAP.items()}


def kind_display_name(item: Item):
    for cls in item.__class__.__mro__[:-2]:  # Spare out object and Item itself
        if cls in _CLASS_TO_NAME_MAP:
            return _CLASS_TO_NAME_MAP[cls]
    return item.category


def _create_instance_test(cls: type[Item]) -> Callable[[Item], bool]:
    def accept(item: Item) -> bool:
        return isinstance(item, cls)
    return accept


def _create_category_test(category: str) -> Callable[[Item], bool]:
    def accept(item: Item) -> bool:
        return item.category == category
    return accept


def filter_by_kind_display_name(items: Iterable[Item], name: str) -> Iterable[Item]:
    """Filters items, either by instance-checking with the class associated with that name in _NAME_TO_CLASS_MAP,
    or, if there is no associate class, by comparing the items' category to the given string."""
    cls = _NAME_TO_CLASS_MAP.get(name)
    accept = _create_instance_test(cls) if cls else _create_category_test(name)
    return (item for item in items if accept(item))


class ItemClassTest(str, enum.Enum):
    Equipment = "Equipment", Equipment
    Armor = "Armor", Armor
    Weapon = "Weapon", Weapon
    Tool = "Tool", Tool
    MountOrVehicle = "Mount or Vehicle", MountOrVehicle
    AdventuringGear = "Adventuring Gear", AdventuringGear
    EquipmentPack = "Equipment Pack", EquipmentPack
    Poison = "Poison", Poison
    MagicItem = "Magic Item", MagicItem
    Spell = "Spell", Spell
    Language = "Language", Language
    Proficiency = "Proficiencies", Proficiency

    FakeEquipment = FakeEquipment
    Immunity = "Immunity"
    Resistance = "Resistance"
    Vulnerability = "Vulnerability"

    AbilityScoreImprovement = "Ability Score Improvement"
    Alignment = "Alignment"
    Archetype = "Archetype"
    ArchetypeFeature = "Archetype Feature"
    # Armor = "Armor"
    ArmorGroup = "Armor Group"
    Background = "Background"
    BackgroundFeature = "Background Feature"
    BackgroundVariant = "Background Variant"
    # Character = "Character"
    Class = "Class"
    ClassFeature = "Class Feature"
    Companion = "Companion"
    CompanionAction = "Companion Action"
    CompanionReaction = "Companion Reaction"
    CompanionTrait = "Companion Trait"
    # Condition = "Condition"
    DamageType = "Damage Type"
    Deity = "Deity"
    Dragonmark = "Dragonmark"
    Feat = "Feat"
    FeatFeature = "Feat Feature"
    Grants = "Grants"
    Ignore = "Ignore"
    Information = "Information"
    Internal = "Internal"
    # Item = "Item"
    # Language = "Language"
    Level = "Level"
    # MagicItem = "Magic Item"
    # MagicSchool = "Magic School"
    Option = "Option"
    # Proficiency = "Proficiency"
    Property = "Property"
    Race = "Race"
    RaceVariant = "Race Variant"
    RacialTrait = "Racial Trait"
    Rule = "Rule"
    Size = "Size"
    Source = "Source"
    # Spell = "Spell"
    SpellcastingFocusGroup = "Spellcasting Focus Group"
    SubRace = "Sub Race"
    Support = "Support"
    Vision = "Vision"
    # Weapon = "Weapon"
    WeaponCategory = "Weapon Category"
    WeaponGroup = "Weapon Group"
    WeaponProperty = "Weapon Property"

    def __new__(cls, name: str, item_class: type[Item] | None = None) -> "ItemClassTest":
        obj = super().__new__(cls, name)
        obj._value_ = name
        return obj

    def __init__(self, name: str, item_class: type[Item] | None = None):
        self._name_ = name
        self._item_class = item_class
        test = _create_instance_test(item_class) if item_class else _create_category_test(name)
        test.__name__ = "test"
        test.__qualname__ = self.__class__.__name__ + "." + test.__name__
        self.test = test

    def __call__(self, item: Item) -> bool:
        return self.test(item)

    def test_for_category(self, category: str) -> Callable[[Item], bool]:
        def test(item: Item) -> bool:
            return self.test(item) and item.category == category
        return test


# ITEM DATA PARSER INTERFACE

def parse_description(info: PreParsing) -> tuple[str, ...]:
    text = []
    for desc_node in info.others.get("description", []):
        inner_text = desc_node.text.strip() if desc_node.text else ""
        if inner_text:
            # Happens with source-kind items form Aurora-Original because they (correctly) use CDATA <![CDATA[...]]>
            text.append(inner_text)
        for child_node in desc_node:
            reference = child_node.attrib.get("element", "")
            if reference:
                text.append(reference)  # typically starts with "ID_"
            elif child_node.text or child_node:  # If it has content
                text.append(ETreeModule.tostring(child_node, "unicode").strip())  # starts with "<"
    return tuple(text)


def inner_xml(node: Element) -> str:
    return (node.text or "").strip() + "".join(ETreeModule.tostring(child, "unicode").strip() for child in node)


ItemT = TypeVar("ItemT", bound=Item)
EquipmentT = TypeVar("EquipmentT", bound=Equipment)


class AbstractAttrsParser(Generic[ItemT], metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def create(cls, info: PreParsing) -> ItemT: ...


class FactoryAttrsParser(Generic[ItemT], AbstractAttrsParser[ItemT]):
    FACTORY: type[ItemT]

    @classmethod
    def create(cls, info: PreParsing) -> ItemT:
        return cls.FACTORY(info.id, info.name, info.source, cls.category(info),
                           cls.description(info), *cls.parse_attrs(info))

    @classmethod
    def category(cls, info: PreParsing) -> str:
        return info.kind

    description = parse_description

    @classmethod
    def parse_attrs(cls, info: PreParsing) -> Sequence:
        return ()


class AbstractEquipmentParser(Generic[EquipmentT], FactoryAttrsParser[EquipmentT], metaclass=ABCMeta):
    FACTORY: type[ItemT]

    @classmethod
    def create(cls, info: PreParsing) -> EquipmentT:
        # Cost
        cost_node = info.setters.get("cost")
        if cost_node is not None and cost_node.text and cost_node.text != "0":
            cost = Quantity.parse_number(cost_node.text, cost_node.attrib["currency"])
        else:
            _logger.info("Missing cost in %s", info.id)
            cost = Quantity()
        # Weight
        if "weight" in info.setters:
            weight = float(info.setters["weight"].attrib["lb"])
        else:
            _logger.info("Missing weight in %s", info.id)
            weight = 0.0
        # Equipment item
        return cls.FACTORY(info.id, info.name, info.source, cls.category(info), cls.description(info),
                           cost, weight, *cls.parse_attrs(info))


class OtherItemParser(FactoryAttrsParser[Item]):
    FACTORY = Item


# CONCRETE EQUIPMENT ATTRS PARSER CLASSES

class ToolParser(AbstractEquipmentParser[Tool]):
    FACTORY = Tool
    NON_ARTISANS_TOOLS = {"Herbalism Kit", "Navigator’s Tools", "Poisoner’s Kit", "Thieves’ Tools"}

    @classmethod
    def category(cls, info: PreParsing) -> str:
        if info.attrs["category"] == "Musical Instruments":
            return "Musical Instrument"
        elif "game" in info.attrs.get("keywords", ""):
            return "Gaming Set"
        elif info.source == "Player’s Handbook" and info.name not in cls.NON_ARTISANS_TOOLS:
            return "Artisan's Tools"
        return "Other Tools"

    @classmethod
    def parse_attrs(cls, info: PreParsing) -> tuple[str]:
        return info.setters["proficiency"].text or "",


class ArmorParser(AbstractEquipmentParser[Armor]):
    FACTORY = Armor

    @classmethod
    def category(cls, info: PreParsing) -> str:
        return info.attrs["armor"]

    @classmethod
    def parse_attrs(cls, info: PreParsing) -> tuple[str, int, int, bool]:
        proficiency = info.setters["proficiency"].text or ""
        ac_desc = info.attrs["armorClass"]
        armor_class = int(ac_desc[:ac_desc.index(" ")] if " " in ac_desc else ac_desc)
        min_strength = int(info.attrs.get("strength", 0))
        stealth_disadv = "stealth" in info.attrs
        return proficiency, armor_class, min_strength, stealth_disadv


class MountOrVehicleParser(AbstractEquipmentParser[MountOrVehicle]):
    FACTORY = MountOrVehicle
    SPEED_PATTERN = re.compile("<strong><em>Speed\\.</em></strong> ?([0-9]+) (ft\\.|mph)")
    CAPACITY_PATTERN = re.compile("<strong><em>Carrying Capacity\\.</em></strong> ?([0-9]+) lb\\.")
    WATERBORNE_VEHICLES = {"Galley", "Keelboat", "Longship", "Rowboat", "Sailing Ship", "Warship"}

    @classmethod
    def category(cls, info: PreParsing) -> str:
        if info.attrs.get("type", "") == "Mount":
            return "Mounts and Other Animals"
        elif info.name in cls.WATERBORNE_VEHICLES:
            return "Waterborne Vehicles"
        return "Tack, Harness, and Drawn Vehicles"

    @classmethod
    def parse_attrs(cls, info: PreParsing) -> tuple[Quantity, int]:
        desc = "".join(ETreeModule.tostring(nodes, "unicode", "html")
                       for nodes in info.others.get("description", ()))
        speed = Quantity.parse_regex_match(cls.SPEED_PATTERN.search(desc))
        capacity_match = cls.CAPACITY_PATTERN.search(desc)
        capacity = int(capacity_match.group(1)) if capacity_match else 0
        return speed, capacity


class AdventuringGearParser(AbstractEquipmentParser[AdventuringGear]):
    FACTORY = AdventuringGear
    CORE_SOURCES = {"Player’s Handbook", "Dungeon Master’s Guide"}

    @classmethod
    def category(cls, info: PreParsing) -> str:
        category = info.attrs["category"]
        if category == "Spellcasting Focus":
            return info.attrs.get("container", category)
        elif category != "Adventuring Gear":
            # Ammunition, Explosives, Treasure
            return category[:-1] if category.endswith("s") else category
        elif info.source in cls.CORE_SOURCES:
            return "Standard Gear"
        return "Other Gear"

    @classmethod
    def parse_attrs(cls, info: PreParsing) -> tuple[Quantity]:
        return cls._gear_item_quantity(info),

    @classmethod
    def _gear_item_quantity(cls, info: PreParsing) -> Quantity:
        # Quantity in parentheses at the end of the name
        quantity_match = re.fullmatch(".* \\((.*)\\)", info.name)
        if not quantity_match:
            return Quantity()  # No quantity at all
        quantity = quantity_match.group(1)
        # "value unit"-scheme
        value_unit_match = re.fullmatch("(one|[0-9,]+)[ \\-—](.*)", quantity)
        if value_unit_match:
            return Quantity.parse_regex_match(value_unit_match)
        # ".*(value).*"-scheme
        value_match = re.search("([0-9,]+)", quantity)
        if value_match:
            return Quantity.parse_number(value_match.group(1), "pieces")
        # "unit"-scheme
        return Quantity(1.0, quantity)


class EquipmentPackParser(AbstractEquipmentParser[EquipmentPack]):
    FACTORY = EquipmentPack

    @classmethod
    def category(cls, info: PreParsing) -> str:
        return "Equipment Pack"

    @classmethod
    def parse_attrs(cls, info: PreParsing) -> tuple[tuple[EquipmentPack.Content, ...]]:
        return cls._equipment_pack_contents(info),

    @classmethod
    def _equipment_pack_contents(cls, info: PreParsing) -> tuple[EquipmentPack.Content, ...]:
        contents = []
        for extract_node in info.others.get("extract", ()):
            for content_node in extract_node:
                if content_node.tag != "item":
                    _logger.warning("Invalid content item '%s' in %s", content_node.tag, info.id)
                    continue
                quantity = Quantity.parse_number(content_node.attrib.get("amount", "1"), "pieces")
                contents.append(EquipmentPack.Content(content_node.text or "", quantity))
        return tuple(contents)


class PoisonParser(AbstractEquipmentParser[Poison]):
    FACTORY = Poison

    @classmethod
    def category(cls, info: PreParsing) -> str:
        alt_name = info.others["sheet"][0].attrib["alt"]
        return alt_name[alt_name.index("(") + 1:-1]


class WeaponParser(AbstractEquipmentParser[Weapon]):
    @staticmethod
    def _weapon_factory(*args) -> Weapon:
        *args, misfire, reload = args
        return Gun(*args, misfire, reload) if misfire or reload else Weapon(*args)

    FACTORY = _weapon_factory

    FIREARM_EXTRA = "ID_WOTC_DMG_WEAPON_PROPERTY_FIREARM_"  # Extra number info
    # FIREARM_NORMAL = "ID_INTERNAL_WEAPON_PROPERTY_FIREARM_"  #   # Unused except for their item definition!
    PROPERTY = "ID_INTERNAL_WEAPON_PROPERTY_"
    GUNSLINGER = "ID_CRIT_WEAPON_PROPERTY_"  # Extra number info
    SPECIAL = "_SPECIAL_"

    LEN_FIREARM_EXTRA = len(FIREARM_EXTRA)
    # LEN_FIREARM_NORMAL = len(FIREARM_NORMAL)  # UNUSED
    LEN_PROPERTY = len(PROPERTY)
    LEN_GUNSLINGER = len(GUNSLINGER)

    # ID_DAMAGE = "ID_INTERNAL_DAMAGE_TYPE_"
    # LEN_DAMAGE = LEN(ID_DAMAGE)
    ID_CATEGORY = "ID_INTERNAL_WEAPON_CATEGORY_"
    LEN_CATEGORY = len(ID_CATEGORY)

    @classmethod
    def category(cls, info: PreParsing) -> str:
        group = ""
        for flag in info.flags:
            if flag.startswith(cls.ID_CATEGORY):
                group = " ".join(map(str.capitalize, flag[cls.LEN_CATEGORY:].split("_")))
                if group != "Martial Ranged":
                    return group  # Note: 'Martial Ranged' may be overwritten by 'Firearm'
        return group

    @classmethod
    def description(cls, info: PreParsing) -> tuple[str, ...]:
        return tuple(filter(lambda line: not line.startswith("ID_"), super().description(info)))

    @classmethod
    def parse_attrs(cls, info: PreParsing) \
            -> tuple[str, tuple[str, ...], tuple[str, ...], Weapon.Damage, Weapon.Range, int, int]:
        properties, links, misfire, reload = cls.properties(info)
        # Damage from setter (ignoring damage identifier in flags)
        damage_node = info.setters.get("damage")
        if damage_node is not None and damage_node.text and damage_node.text != "—":
            ddice = Dice.parse(damage_node.text)
            dtype = damage_node.attrib["type"]
        else:
            ddice = Dice()
            dtype = ""
        # Special case: Lance isn't versatile but has the versatile property set!
        versatile = Dice.parse(info.attrs.get("versatile", "")) if "Versatile" in properties else Dice()
        damage = Weapon.Damage(dtype, ddice, versatile)
        # Range
        normal, long = map(int, info.attrs["range"].split("/")) if "range" in info.attrs else (0, 0)
        weapon_range = Weapon.Range(normal, long)
        # Finalize
        proficiency = info.setters["proficiency"].text or ""
        return proficiency, properties, links, damage, weapon_range, misfire, reload

    @classmethod
    def properties(cls, info: PreParsing) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
        properties = dict[str, str]()
        misfire = shots = 0
        for flag in info.flags:
            # Firearm property with extra info
            if flag.startswith(cls.FIREARM_EXTRA):
                parts = flag[cls.LEN_FIREARM_EXTRA:].split("_")
                name = parts[0].capitalize()
                if name == "Reload" and len(parts) > 1:
                    flag = cls.FIREARM_EXTRA + parts[0]  # Cut off extra number
                    shots = int(parts[1])
                elif name == "Burst":
                    name += " Fire"
            # Firearm property without extras - UNUSED
            # elif flag.startswith(cls.FIREARM_NORMAL):
            #     name = flag[cls.LEN_FIREARM_NORMAL:].capitalize()
            # Special weapon property
            elif cls.SPECIAL in flag:
                name = "Special"
                if flag == "ID_INTERNAL_WEAPON_PROPERTY_SPECIAL_CHAKRAM":
                    flag = "ID_AW_OOTD_WEAPON_PROPERTY_SPECIAL_CHAKRAM"
            # Standard weapon property
            elif flag.startswith(cls.PROPERTY):
                name = flag[cls.LEN_PROPERTY:].capitalize()
                if name == "Twohanded":
                    name = "Two-Handed"
                elif name == "Special":
                    continue
            # Gunslinger weapon property
            elif flag.startswith(cls.GUNSLINGER):
                parts = flag[cls.LEN_GUNSLINGER:].split("_")
                name = parts[0].capitalize()
                if len(parts) > 1:
                    flag = cls.GUNSLINGER + parts[0]  # Cut off extra number
                    if name == "Misfire":
                        misfire = int(parts[1])
                    elif name == "Reload":
                        shots = int(parts[1])
            else:
                continue
            properties[name] = flag
        # Sort properties by name before splitting the dict up
        properties = {name: properties[name] for name in sorted(properties)}
        return tuple(properties.keys()), tuple(properties.values()), misfire, shots


class FakeEquipmentParser(FactoryAttrsParser[FakeEquipment]):
    FACTORY = FakeEquipment

    @classmethod
    def category(cls, info: PreParsing) -> str:
        return info.attrs["category"]


# OTHER CONCRETE FACTORY ATTRS PARSER

class SpellParser(FactoryAttrsParser[Spell]):
    FACTORY = Spell

    # Boolean fields and material
    BOOL_KEYS = "isConcentration", "isRitual", "hasVerbalComponent", "hasSomaticComponent"
    COST_PATTERN = re.compile("([0-9,]*[0-9]) (gp|sp|cp|ep|pp)")

    # Casting time
    TIME_PATTERN = re.compile("([0-9]+) (actoin|action|bonus action|reaction|"
                              "round|rounds|minute|minutes|min\\.|hour|hours)")
    TIME_MAP = {"actoin": "action", "min.": "minute"}  # , "minutes": "minute", "hours": "hour"}
    SPECIAL_STARTS = "which you take when ", "which you take after ", "which you take in response to ", \
        "which you take ", "taken when ", "taken ", "made when ", "made ", "when ", "in response to "

    # Range and area of effect
    DISTANCE_PATTERN = re.compile("([0-9,]+) (ft\\.?|feet|Feet|miles?)")
    DISTANCE_AOE_PATTERN = re.compile("(Self|Touch) \\(([^)]*)\\)?")
    AOE_PATTERN = re.compile("([0-9]+)[- ](foot|feet)[- ]?(radius)? ?(.+)?")  # Area of Effect

    # Duration
    CONCENTRATION_PATTERN = re.compile("Con?centration, (.*)")
    # DURATION_SPLIT = re.compile(", or |, | or |; ")
    DURATION_PATTERN = re.compile("(up to )?([0-9]+) ([A-Z]*)\\.?|until (.*)", re.IGNORECASE)
    DURATION_MAP = {"Variable": "Special", "See below": "Special",
                    "Intantaneous": "Instantaneous", "Instantaenous": "Instantaneous"}

    @classmethod
    def category(cls, info: PreParsing) -> str:
        return info.attrs["school"].capitalize()

    @classmethod
    def parse_attrs(cls, info: PreParsing) -> tuple[int, bool, bool, bool, bool, str, Quantity, Quantity, str,
                                                     Quantity, Quantity, Quantity, bool]:
        level = int(info.attrs["level"])
        con, rit, verbal, somatic = cls._spell_boolean_values(info.attrs, info.id)
        material, cost = cls._spell_material(info.attrs, info.id)
        time, desc = cls._casting_time(info.attrs["time"])
        if "range" in info.attrs:
            spell_range, aoe = cls._spell_range(info.attrs["range"], info.id)
        else:
            _logger.warning("Missing spell range in %s", info.id)
            spell_range = aoe = Quantity()
        duration, flag = cls._spell_duration(info.attrs["duration"])
        return level, con, rit, verbal, somatic, material, cost, time, desc, spell_range, aoe, duration, flag

    @classmethod
    def _bool_value(cls, text: str) -> bool:
        return text.strip().lower() == "true"

    @classmethod
    def _spell_boolean_values(cls, setters: dict[str, str], aurora_id: str) -> list[bool]:
        values = list[bool]()
        for key in cls.BOOL_KEYS:
            if key in setters:
                values.append(cls._bool_value(setters[key]))
            else:
                _logger.info("Missing '%s' setter in %s", key, aurora_id)
                values.append(False)
        return values

    @classmethod
    def _spell_material(cls, setters: dict[str, str], item_id: str) -> tuple[str, Quantity]:
        material, has_material = setters.get("materialComponent", ""), setters.get("hasMaterialComponent", "")
        material = material if material not in ("V", "false", None) else ""
        # Inspect the material component
        if material and not cls._bool_value(has_material):
            _logger.warning("Material component mismatch in %s: False vs. '%s'", item_id, material)
        # Material costs
        cost = Quantity()
        cost_match = cls.COST_PATTERN.findall(material)
        if cost_match:
            currencies = {"pp": 1000, "gp": 100, "ep": 50, "sp": 10, "cp": 1}
            cost_int = sum(Quantity.parse_number(*cost).unitless(currencies) for cost in cost_match)
            for unit, valence in currencies.items():
                if cost_int % valence == 0:
                    cost = Quantity(cost_int // valence, unit)
                    break
            _logger.debug("Spell %s has costs %s (total: %s)", item_id, cost_match, cost)
        return material, cost

    @classmethod
    def _casting_time(cls, casting_time: str) -> tuple[Quantity, str]:
        # 1. Split into primary casting time and additional time specification
        time_spec = ""
        # Reactions that have a specification of when to take them
        if casting_time.startswith("1 reaction, "):
            time_spec = casting_time[12:]
            casting_time = "1 reaction"
            for start in cls.SPECIAL_STARTS:
                if time_spec.startswith(start):
                    time_spec = time_spec[len(start):]
                    break
        # Strip of details that are not further specified
        elif casting_time.endswith(" (see below)"):
            casting_time = casting_time[:-12]
        # Move additional, alternative casting times to the specs
        multiple = casting_time.split(" or ")
        if len(multiple) > 1:
            time_spec += (" / " if time_spec else "") + " / ".join(multiple[1:])
        value = multiple[0]

        # 2. Parse casting time according to value-unit scheme
        pattern_match = cls.TIME_PATTERN.fullmatch(value.lower())
        if pattern_match:
            number, unit = pattern_match.groups()
            unit = cls.TIME_MAP.get(unit, unit)
            return Quantity.parse_number(number, unit), time_spec
        return Quantity(0.0, "Special" if value == "Variable" else value), time_spec

    @classmethod
    def _spell_range(cls, spell_range: str, aurora_id: str) -> tuple[Quantity, Quantity]:
        dist_match = cls.DISTANCE_PATTERN.fullmatch(spell_range)
        if dist_match:
            return Quantity.parse_number(*dist_match.groups()), Quantity()
        dist_aoe_match = cls.DISTANCE_AOE_PATTERN.fullmatch(spell_range)
        if dist_aoe_match:
            aoe_ranges = cls._area_of_effect(dist_aoe_match.group(2), aurora_id)
            return Quantity(0.0, dist_aoe_match.group(1)), aoe_ranges
        aoe_ranges = cls._area_of_effect(spell_range)
        if aoe_ranges:
            return Quantity(), aoe_ranges
        elif re.fullmatch("[0-9]+", spell_range):
            return Quantity.parse_number(spell_range, "ft"), Quantity()
        range_map = {"Personal": "Self", "Variable": "Special"}
        return Quantity(0.0, range_map.get(spell_range, spell_range)), Quantity()

    @classmethod
    def _area_of_effect(cls, spell_range: str, aurora_id: str = "") -> Quantity:
        aoe_match = cls.AOE_PATTERN.match(spell_range)
        if aoe_match:
            distance, feet, radius, shape = aoe_match.groups()
            geometry = shape or radius or "distance"
            return Quantity.parse_number(distance, geometry)
        if aurora_id:
            _logger.warning("Not matching AOE pattern in %s: %s", aurora_id, spell_range)
        return Quantity()

    @classmethod
    def _spell_duration(cls, duration: str) -> tuple[Quantity, bool]:
        if duration == "Instantaneous or 1 hour (see below)":
            return Quantity(0.0, "Instantaneous"), False
        con_match = cls.CONCENTRATION_PATTERN.fullmatch(duration)
        if con_match:
            duration = con_match.group(1)
        duration_match = cls.DURATION_PATTERN.fullmatch(duration)
        if duration_match:
            up_to, value, unit, until = duration_match.groups()
            return Quantity(0.0, until) if until \
                       else Quantity.parse_number(value, unit), bool(up_to or until)
        return Quantity(0.0, cls.DURATION_MAP.get(duration, duration)), False


class MagicItemParser(FactoryAttrsParser[MagicItem]):
    FACTORY = MagicItem
    CATEGORIES = {"Armor", "Weapon", "Wondrous Item", "Scroll", "Potion", "Ring", "Rod", "Staff", "Wand",
                  "Supernatural Gift"}
    CATEGORY_MAP = {variant.lower(): name for name in CATEGORIES for variant in (name, name + "s")}
    RARITY_SPELLINGS = {"lgendary": "legendary", "vert rare": "very rare", "infusion": "artificer infusion"}
    RARITIES = {"common", "uncommon", "rare", "very rare", "legendary", "artifact", "unique",
                "artificer infusion", "varying", ""}

    @classmethod
    def create(cls, info: PreParsing) -> ItemT:
        category = cls.category(info)
        if category == "Supernatural Gift":
            return Item(info.id, info.name, info.source, category, cls.description(info))
        return cls.FACTORY(info.id, info.name, info.source, category,
                           cls.description(info), *cls.parse_attrs(info))

    @classmethod
    def category(cls, info: PreParsing) -> str:
        for name in "type", "category":
            value = cls.CATEGORY_MAP.get(info.attrs.get(name, "").lower())
            if value:
                return value
        return ""

    @classmethod
    def parse_attrs(cls, info: PreParsing) -> tuple[str, str, str]:
        spec = info.setters["type"].attrib.get("addition", "") if "type" in info.setters else ""
        rarity = info.attrs.get("rarity", "").lower()
        rarity = cls.RARITY_SPELLINGS.get(rarity, rarity)
        rarity = rarity if rarity in cls.RARITIES else "varying"
        attune = cls._attunement(info.setters.get("attunement"))
        return spec, rarity, attune

    @classmethod
    def _attunement(cls, setter: Element | None) -> str:
        if setter is not None and setter.text and setter.text.strip().lower() == "true":
            attune = setter.attrib.get("addition")
            if not attune:
                return "anyone"
            elif attune.startswith("by "):
                return attune[3:]
            return attune
        return ""


class LanguageParser(FactoryAttrsParser[Language]):
    FACTORY = Language

    @classmethod
    def category(cls, info: PreParsing) -> str:
        for category in "Standard", "Exotic", "Secret", "Monster":
            if category in info.flags:
                return category
        return ""

    @classmethod
    def parse_attrs(cls, info: PreParsing) -> tuple[str, str]:
        return cls._prop_text(info, "script"), cls._prop_text(info, "speakers")

    @classmethod
    def _prop_text(cls, info: PreParsing, prop_name: str) -> str:
        prop_text = info.attrs.get(prop_name, "")
        return "" if prop_text == "—" else prop_text


class PlayerClassParser(FactoryAttrsParser[PlayerClass]):
    FACTORY = PlayerClass

    @classmethod
    def parse_attrs(cls, info: PreParsing) -> tuple[int, str]:
        return int(info.attrs["hd"][1:]), info.attrs["short"]


class LevelParser(FactoryAttrsParser[Level]):
    FACTORY = Level

    @classmethod
    def parse_attrs(cls, info: PreParsing) -> tuple[int, int]:
        return int(info.attrs["Level"]), int(info.attrs["Experience"])


class FeatureParser(FactoryAttrsParser[Feature]):
    FACTORY = Feature

    @classmethod
    def parse_attrs(cls, info: PreParsing) -> tuple[str, str, str, tuple[tuple[int, str], ...]]:
        sheet_nodes = info.others.get("sheet")
        if not sheet_nodes:
            return "", "", "", ()
        sheet_node = sheet_nodes[0]
        attrs = sheet_node.attrib
        title = attrs.get("alt", "") or attrs.get("name", "")
        usage = attrs.get("usage", "")
        action = attrs.get("action", "")
        short: dict[int, str] = {int(desc_node.attrib.get("level", 1)): inner_xml(desc_node)
                                 for desc_node in sheet_node if desc_node.text or desc_node}
        final_short = tuple((level, short[level]) for level in sorted(short))
        return title, usage, action, final_short


# OTHER CONCRETE ATTRS PARSERS

class ProficiencyParser(AbstractAttrsParser[Proficiency]):
    _PROFICIENCY = " Proficiency ("
    _LEN_PROFICIENCY = len(_PROFICIENCY)
    _FOCUS_CATEGORY = "Spellcasting Focus"
    _FOCUS_START = _FOCUS_CATEGORY + " Group ("
    _LEN_FOCUS = len(_FOCUS_START)

    class _SkillException(Exception):
        pass

    @classmethod
    def create(cls, info: PreParsing) -> ItemT:
        desc = parse_description(info)
        try:
            name, category = cls.name_and_category(info)
            return Proficiency(info.id, name, info.source, category, desc)
        except cls._SkillException:
            return Skill(info.id, info.name, info.source, "Skill", desc, info.flags[1])

    @classmethod
    def name_and_category(cls, info: PreParsing) -> tuple[str, str]:
        name = info.name
        index = name.find(cls._PROFICIENCY)
        if index >= 0:  # Pattern "<Category> Proficiency (<Name>)"
            return name[index + cls._LEN_PROFICIENCY:-1], name[:index]
        elif name.startswith(cls._FOCUS_START):
            return name[cls._LEN_FOCUS:-1], cls._FOCUS_CATEGORY
        raise cls._SkillException()  # Special case: skill proficiency


class WeaponPropertyParser(AbstractAttrsParser[WeaponProperty]):
    _parentheses = re.compile("(.*) \\((.*)\\)")
    _tail_number = re.compile("(.*) ([0-9]+)")
    _number_unit = re.compile("([0-9]+) (.*)")

    @classmethod
    def create(cls, info: PreParsing) -> WeaponProperty:
        match_parentheses = cls._parentheses.match(info.name)
        if match_parentheses:
            name, parentheses = match_parentheses.groups()
            match_number_unit = cls._number_unit.match(parentheses)
            if match_number_unit:
                value = Quantity.parse_number(*match_number_unit.groups())
            else:
                value = Quantity.text_only(parentheses)
        else:
            match_tail_number = cls._tail_number.match(info.name)
            if match_tail_number:
                name, number = match_tail_number.groups()
                value = Quantity.parse_number(number)
            else:
                name = info.name
                value = Quantity()
        return WeaponProperty(info.id, name, info.source, info.kind, parse_description(info), value)


class FeatureExpertiseParser(AbstractAttrsParser[Item]):
    """Expertise is in fact a class feature!"""

    EXPERTISE_ID = "ID_EXPERTISE_SKILL_"
    EXPERTISE_START = "Skill Expertise ("
    EXPERTISE_START_LEN = len(EXPERTISE_START)
    EXPERTISE_CATEGORY = "Expertise"

    @classmethod
    def create(cls, info: PreParsing) -> Item:
        text = parse_description(info)
        if info.name.startswith(cls.EXPERTISE_START):
            name = info.name[cls.EXPERTISE_START_LEN:-1]
            return Item(info.id, name, info.source, cls.EXPERTISE_CATEGORY, text)
        return FeatureParser.create(info)


class WeaponGroupParser(AbstractAttrsParser[Item]):
    _WEAPON_GROUP = "Weapon Group"
    _WG_START_LEN = len(_WEAPON_GROUP) + 2

    @classmethod
    def create(cls, info: PreParsing) -> Item:
        name = info.name[cls._WG_START_LEN:-1]
        text = parse_description(info)
        return Item(info.id, name, info.source, info.kind, text)


class ConditionParser(AbstractAttrsParser[Condition]):
    _PATTERN = re.compile("(.*) \\((.*)\\)")

    @classmethod
    def create(cls, info: PreParsing) -> ItemT:
        what, against = cls._PATTERN.match(info.name).groups()
        text = parse_description(info)
        return Condition(info.id, against, info.source, what, text)


# GENERAL ITEM DATA PARSING

class EquipmentParser(AbstractAttrsParser[Item]):
    parser: dict[str, AbstractAttrsParser] = {
        "Mounts & Vehicles": MountOrVehicleParser,
        "Tools": ToolParser,
        "Musical Instruments": ToolParser,
        "Supernatural Gifts": MagicItemParser,
        "Equipment Packs": EquipmentPackParser,
        "Poison": PoisonParser,
        "Additional Feature": FakeEquipmentParser,
        "Additional Ability Score Improvement": FakeEquipmentParser,
        "Optional Class Features": FakeEquipmentParser,
    }
    gear_parser = AdventuringGearParser

    @classmethod
    def create(cls, info: PreParsing) -> Item:
        return cls.parser.get(info.attrs["category"], cls.gear_parser).create(info)


class ItemAttrsParser(AuroraPostParser[Item]):
    parser: dict[str, AbstractAttrsParser] = {
        "Item": EquipmentParser,
        "Armor": ArmorParser,
        "Weapon": WeaponParser,
        "Magic Item": MagicItemParser,
        "Spell": SpellParser,
        "Language": LanguageParser,
        "Proficiency": ProficiencyParser,
        "Level": LevelParser,
        "Class": PlayerClassParser,
        "Feat": FeatureParser,
        "Feat Feature": FeatureParser,
        "Archetype Feature": FeatureParser,
        "Racial Trait": FeatureParser,
        "Class Feature": FeatureExpertiseParser,
        "Condition": ConditionParser,
        "Weapon Group": WeaponGroupParser,
        "Weapon Property": WeaponPropertyParser,
    }
    default = OtherItemParser

    @classmethod
    def create(cls, info: PreParsing) -> Item:
        return cls.parser.get(info.kind, cls.default).create(info)

    _OUTPUT_TYPE = Item
    _DYNAMIC_MODELS = Item,


# MAIN - some test programs

if __name__ == '__main__':
    import logging
    from pprint import pprint

    from auroratools.core.utils import configure_logging
    from auroratools.core.base import index_aurora_files

    configure_logging(logging.DEBUG)

    BASE = Path("../../../")
    XML_INDEX = index_aurora_files(BASE / "../content")
    INFO_INDEX = PreParsing.transform_index(XML_INDEX)


    def print_node_main_tags():
        child_tags = defaultdict(int)
        for node_id, nodes in XML_INDEX.items():
            for node in nodes:
                for child_node in node:
                    child_tags[child_node.tag] += 1
        pprint(child_tags, width=160)


    def weapons_weapon_properties():
        props = set[str]()
        for node_id, nodes in XML_INDEX.items():
            info = PreParsing.create(nodes)
            if info.kind == "Weapon":
                props.update(info.flags)
        pprint(sorted(props), width=160)


    def test_data_parsing():
        item_collection = defaultdict[tuple[str, str], list[Item]](list)
        for info in INFO_INDEX.values():
            item = ItemAttrsParser.create(info)
            item_collection[(info.kind, item.__class__.__name__)].append(item)

        for (aurora_type, item_class), sub_items in item_collection.items():
            if aurora_type == "Item" or item_class != "Item":
                print("\n#", aurora_type, "-", item_class, "({})".format(len(sub_items)))
                pprint(sorted(set(item.category for item in sub_items)), width=160)
                pprint(sub_items[:7], width=250)

    def items_with_proficiency():
        groups = defaultdict(list)
        for item_info in INFO_INDEX.values():
            if "proficiency" in item_info.setters:
                groups[item_info.kind].append(item_info)
        pprint(sorted(groups.keys()))
        pprint({k: len(v) for k, v in groups.items()})
        print("Total:", sum(map(len, groups.values())))
        pprint({k: [(x.name, x.setters["proficiency"].text) for x in v]
                for k, v in groups.items()}, width=160)

    def item_names_with_parentheses():
        pattern = re.compile("(.*) \\((.*)\\)")
        total_counts = defaultdict(int)
        par_counts = defaultdict(int)
        col_counts = defaultdict(int)
        for item in INFO_INDEX.values():
            if ("(" in item.name or ")" in item.name) and not pattern.match(item.name):
                print("Inconsistent:", item.id, item.name)
            match = pattern.match(item.name)
            if match:
                par_counts[item.kind] += 1
            total_counts[item.kind] += 1
            if ":" in item.name:
                col_counts[item.kind] += 1
        print("\n# PARENTHESES")
        pprint(sorted({t: (c, par_counts[t], par_counts[t] / c)
                       for t, c in total_counts.items()}.items(), key=lambda x: -x[1][-1]), width=160)
        print("\n# COLON")
        pprint(sorted({t: (c, col_counts[t], col_counts[t] / c)
                       for t, c in total_counts.items()}.items(), key=lambda x: -x[1][-1]), width=160)

    def sheet_counts():
        for info in INFO_INDEX.values():
            if "sheet" not in info.others:
                _logger.warning("%s: no sheet", info.id)
            elif len(info.others["sheet"]) > 1:
                _logger.warning("%s: multiple sheets", info.id)

    @dataclass(frozen=True)
    class ItemMeta:
        attrs: Item
        flags: set[str]
        kind: str
        # Information to display on a character sheet
        desc: tuple[str] = field(repr=False)
        title: str = ""  # alternative name
        usage: str = ""  # may contain placeholders
        action: str = ""
        show: bool = False
        short: dict[int, str] = field(default_factory=dict, repr=False)

        @classmethod
        def create(cls, info: PreParsing, item: Item) -> "ItemMeta":
            sheet_nodes = info.others.get("sheet")
            if sheet_nodes:
                sheet_args = cls._parse_sheet_node(sheet_nodes[0])
            else:
                sheet_args = ()
            desc = parse_description(info)
            return cls(item, set(info.flags), info.kind, desc, *sheet_args)

        @classmethod
        def _parse_sheet_node(cls, sheet_node: Element) -> tuple[str, str, str, bool, dict[int, str]]:
            attrs = sheet_node.attrib
            title = attrs.get("alt") or attrs.get("name") or ""
            usage = attrs.get("usage", "")
            action = attrs.get("action", "")
            show = attrs.get("display", "").strip().lower() != "false"
            short = {int(desc_node.attrib.get("level", 1)): cls._inner_xml(desc_node)
                     for desc_node in sheet_node if desc_node.text or desc_node}
            return title, usage, action, show, short

        @classmethod
        def _inner_xml(cls, node: Element) -> str:
            return (node.text or "").strip() \
                + "".join(ETreeModule.tostring(child, "unicode").strip() for child in node)

    def test_meta_parsing():
        meta_index = {k: ItemMeta.create(v, ItemAttrsParser.create(v)) for k, v in INFO_INDEX.items()}

        # pprint(sorted(meta_index.values(), key=lambda meta: meta.attrs.id)[:25], width=160)

        ## Long descriptions ##
        # meta_with_desc = filter(lambda meta: len(meta.desc) > 5, meta_index.values())
        # pprint(sorted(((meta.attrs.id, meta.desc) for meta in meta_with_desc),
        #               key=lambda thing: thing[0])[:25], width=160)

        ## Short descriptions ##
        meta_with_short = tuple(filter(lambda meta: len(meta.short) == 1 and set(meta.short).pop() != 1, meta_index.values()))
        pprint(sorted(((meta.attrs.id, meta.short) for meta in meta_with_short))[:50], width=160)
        print("#", len(meta_with_short))
        # 5864 x single short at default level 1
        # 3 x single short at other levels
        # 260 x multiple short

    def text_attr_per_type():
        meta_index: dict[str, ItemMeta] = {k: ItemMeta.create(v, ItemAttrsParser.create(v))
                                           for k, v in INFO_INDEX.items()}
        things = defaultdict(lambda: defaultdict(int))
        for meta in meta_index.values():
            sub = things[meta.kind]
            sub["count"] += 1
            if meta.desc:
                sub["desc"] += 1
            if len(meta.short) == 1 and set(meta.short).pop() == 1:  # default short
                sub["single"] += 1
            elif meta.short:
                sub["multiple"] += 1
            if meta.title:
                sub["title"] += 1
            if meta.usage:
                sub["usage"] += 1
            if meta.action:
                sub["action"] += 1
            if meta.show:
                sub["show"] += 1
        fields = "desc", "single", "multiple", "title", "usage", "action", "show"
        print(" " * 25, "count", *["| {:12} ".format(name) for name in fields])
        for kind in sorted(things, key=lambda x: (- things[x].get("show", 0) / things[x]["count"], x)):
            sub = things[kind]
            x = ["| {:5.2f} ({:5d})".format(sub[name] / sub["count"], sub[name])
                 if name in sub else "|" + " " * 14 for name in fields]
            print("{:25}".format(kind), "{:5d}".format(sub["count"]), *x)
        total = {name: sum(sub[name] for sub in things.values()) for name in ("count", *fields)}
        x = ["| {:5.2f} ({:5d})".format(total[name] / total["count"], total[name])
             if name in total else "| " + " " * 12 for name in fields]
        print("{:25}".format("TOTAL"), "{:5d}".format(total["count"]), *x)

        """
        Armor, Weapon, Item: have desc only and all but some Weapons have one (the few Item-titles are safe to ignore)
        """

    # No multi-desc items :)
    # for xml in INFO_INDEX.values():
    #     if "description" in xml.others and len(xml.others["description"]) > 1:
    #         print(xml.id)

    print_node_main_tags()
    weapons_weapon_properties()
    test_data_parsing()
    items_with_proficiency()
    item_names_with_parentheses()
    sheet_counts()
    test_meta_parsing()
    text_attr_per_type()
