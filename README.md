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
import logging
from pprint import pprint

from auroratools.core.simple_loading import ItemQueryingData
from auroratools.core.utils import configure_logging, track_progress_stdout

configure_logging(logging.ERROR)

items = ItemQueryingData.load_aurora_content_with_caching(
    # Put here the path to directories containing aurora content xml files
    "Path to xml content files directory",
    # The cache file is created when this piece of code is first executed
    cache_file="aurora_content_cache.json",
    progress=track_progress_stdout
)
print("# elements:", len(items))
pprint(list(items.values())[:7], width=180)

engine = ItemQueryingEngine(items.values())
group = input("Type in an item type (e.g. 'Spell'): ")
query = input("Type in a query (e.g. 'Warlock,0|1'): ")
results = list(engine.scan_parse_select_attrs(group, query))
print("# queried items:", len(results))
pprint(results[:7], width=180)
```

To get the content files, see
- [Aurora Legacy Project](https://github.com/AuroraLegacy/elements)
- [Aurora Builder - Additional Content](https://aurorabuilder.com/content/)
