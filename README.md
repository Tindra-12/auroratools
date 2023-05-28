# Aurora tools

A python toolset to parse the xml content files of the Aurora character builder.
In the future, it might also include some useful Qt GUI applications.

## Installation and Requirements

This project requires at least Python 3.10. For now, it has no extra dependencies.

```bash
pip install .
```

## Usage

```python
from auroratools.core.attrs_parsing import ItemAttrsParser

item_index = ItemAttrsParser.load_index_with_caching(
    "out/items_cache.json", "Path to xml content files directory")
```

To get the content files, see
- [Aurora Legacy Project](https://github.com/AuroraLegacy/elements)
- [Aurora Builder - Additional Content](https://aurorabuilder.com/content/)
