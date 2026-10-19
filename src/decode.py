"""Run the trained CRNN over val+test cells and dump CTC n-best (string, log-prob) to data/nbest.json."""
import sys, json, pathlib, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from htr import CRNN, prep, beam_search

ROOT = pathlib.Path(__file__).resolve().parent.parent
torch.set_num_threads(8)
d = np.load(ROOT / "data" / "cells.npz")
model = CRNN(); model.load_state_dict(torch.load(ROOT / "data" / "crnn.pt")); model.eval()
idx = np.where(d["split"] != "train")[0]
out = {}
with torch.no_grad():
    for i in range(0, len(idx), 256):
        b = idx[i:i + 256]
        lp = model(prep(d["X"][b])).numpy()
        for j, k in enumerate(b):
            out[str(d["ids"][k])] = [[s, round(float(l), 4)] for s, l in beam_search(lp[j])]
(ROOT / "data" / "nbest.json").write_text(json.dumps(out))
print("decoded", len(out))
