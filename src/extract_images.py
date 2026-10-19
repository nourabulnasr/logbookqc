"""Stream HisClima.tar.gz once, keep only numeric manuscript cell images (grayscale, height-normalised)."""
import sys, tarfile, io, pathlib, numpy as np
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cells import parse_cells, split_of, RAW

H, MAXW = 32, 96
ROOT = RAW.parent
cells = [c for c in split_of(parse_cells()) if c["split"]]
want = {f"./data/feats/{c['id']}.png": c for c in cells}
imgs = {}
with tarfile.open(RAW / "HisClima.tar.gz", "r|gz") as t:
    for m in t:
        c = want.get(m.name)
        if c is None:
            continue
        im = Image.open(io.BytesIO(t.extractfile(m).read())).convert("L")
        w = max(8, round(im.width * H / im.height))
        im = im.resize((min(w, MAXW), H), Image.BILINEAR)
        a = np.full((H, MAXW), 255, np.uint8)  # right-pad with paper-white
        a[:, : im.width] = np.asarray(im)
        imgs[c["id"]] = (a, im.width)
        if len(imgs) % 2000 == 0:
            print(len(imgs), "/", len(want), flush=True)
missing = [c["id"] for c in cells if c["id"] not in imgs]
print("extracted", len(imgs), "missing", len(missing))
keep = [c for c in cells if c["id"] in imgs]
np.savez_compressed(
    ROOT / "data" / "cells.npz",
    ids=np.array([c["id"] for c in keep]),
    X=np.stack([imgs[c["id"]][0] for c in keep]),
    W=np.array([imgs[c["id"]][1] for c in keep]),
    text=np.array([c["text"] for c in keep]),
    split=np.array([c["split"] for c in keep]),
    page=np.array([c["page"] + "|" + c["table"] for c in keep]),
    c=np.array([c["c"] for c in keep]),
)
