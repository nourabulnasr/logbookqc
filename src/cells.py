"""Parse HisClima transcriptions + split lists into a table of numeric manuscript cells."""
import re, json, collections, pathlib

RAW = pathlib.Path(__file__).resolve().parent.parent / "raw"
ID_RE = re.compile(r"(?P<page>.+?)\.(?P<table>table\d+)\.table\d+_c(?P<c>\d+)_l(?P<l>\d+)$")
NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def load_split(name):
    """Set of table-cell ids in a split (tables only: *_0 lists)."""
    return {l.strip() for l in (RAW / f"{name}_0.lst").read_text().splitlines() if l.strip()}


def parse_cells():
    """Returns list of dict(id,page,table,c,text) for single-token manuscript numeric cells."""
    lines_per_cell = collections.Counter()
    rows = []
    for l in (RAW / "transcriptions.txt").read_text(encoding="utf8").splitlines():
        cid, _, txt = l.partition(" ")
        m = ID_RE.match(cid)
        if not m:
            continue
        lines_per_cell[(m["page"], m["table"], int(m["c"]))] += 1
        rows.append((cid, m, txt))
    out = []
    for cid, m, txt in rows:
        key = (m["page"], m["table"], int(m["c"]))
        toks = txt.split(" ")
        if lines_per_cell[key] != 1 or len(toks) != 1:
            continue
        w, _, tag = toks[0].rpartition("!")
        if tag != "manuscript" or not NUM_RE.match(w):
            continue
        out.append(dict(id=cid, page=m["page"], table=m["table"], c=int(m["c"]), text=w))
    return out


def split_of(cells):
    sp = {n: load_split(n) for n in ("train", "val", "test")}
    for c in cells:
        c["split"] = next((n for n in sp if c["id"] in sp[n]), None)
    return cells


if __name__ == "__main__":
    cs = split_of(parse_cells())
    print(len(cs), collections.Counter(c["split"] for c in cs))
    pages = collections.defaultdict(set)
    for c in cs:
        pages[c["page"]].add(c["split"])
    print("pages in >1 split:", sum(len(v) > 1 for v in pages.values()), "of", len(pages))
    fam = collections.Counter(c["page"].split("_")[0] for c in cs)
    print(fam)
    print(collections.Counter(len(c["text"]) for c in cs).most_common(8))


def table_widths():
    """(page,table) -> column count W, inferred from the printed hour labels ('1','2',...):
    consecutive hours sit exactly W cells apart. Printed form text, not handwriting."""
    pr = collections.defaultdict(dict)
    for l in (RAW / "transcriptions.txt").read_text(encoding="utf8").splitlines():
        cid, _, txt = l.partition(" ")
        m = ID_RE.match(cid)
        if m and m["l"] == "1" and re.fullmatch(r"\d+!print", txt):
            pr[(m["page"], m["table"])][int(m["c"])] = int(txt.split("!")[0])
    out = {}
    for k, d in pr.items():
        by = collections.defaultdict(list)
        for c, v in d.items():
            by[v].append(c)
        diffs = collections.Counter(c2 - c for v, cl in by.items() for c in cl for c2 in by.get(v + 1, []) if 0 < c2 - c < 60)
        if diffs:
            out[k] = diffs.most_common(1)[0][0]
    return out
