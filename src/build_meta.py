"""Write data/cells_meta.json: every numeric manuscript cell with split, layout width W and family."""
import json, sys, pathlib, collections
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cells import parse_cells, split_of, table_widths

ROOT = pathlib.Path(__file__).resolve().parent.parent
cells = [c for c in split_of(parse_cells()) if c["split"]]
tw = table_widths()
fam_default = {}
for fam in ("vol003", "Albatross"):
    fam_default[fam] = collections.Counter(w for (p, t), w in tw.items() if p.startswith(fam)).most_common(1)[0][0]
n_fallback = 0
for c in cells:
    c["fam"] = c["page"].split("_")[0]
    w = tw.get((c["page"], c["table"]))
    if w is None:
        w, n_fallback = fam_default[c["fam"]], n_fallback + 1
    c["W"] = w
(ROOT / "data" / "cells_meta.json").write_text(json.dumps(cells))
print(len(cells), "cells;", n_fallback, "used family-default width;", fam_default)
