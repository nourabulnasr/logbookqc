"""Train the CRNN on train-split numeric cells; early-stop on val CER; write n-best for val+test."""
import sys, json, time, pathlib, numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from htr import CRNN, ALPHA, prep, beam_search

ROOT = pathlib.Path(__file__).resolve().parent.parent
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
d = np.load(ROOT / "data" / "cells.npz")
X, text, split = d["X"], d["text"], d["split"]
tr, va = np.where(split == "train")[0], np.where(split == "val")[0]
enc = lambda s: [ALPHA.index(ch) + 1 for ch in s]


def augment(xb):
    B = xb.shape[0]
    out = torch.empty_like(xb)
    for i in range(B):
        s = np.random.randint(-6, 7)
        out[i] = torch.roll(xb[i], s, dims=-1)
    out = out * (0.8 + 0.4 * torch.rand(B, 1, 1, 1))
    return (out + 0.03 * torch.randn_like(out)).clamp(0, 1)


def ed(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        q = [i]
        for j, y in enumerate(b, 1):
            q.append(min(p[j] + 1, q[j - 1] + 1, p[j - 1] + (x != y)))
        p = q
    return p[-1]


def greedy(lp):
    out = []
    for seq in lp.argmax(-1).numpy():
        s, prev = [], 0
        for c in seq:
            if c != prev and c != 0: s.append(ALPHA[c - 1])
            prev = c
        out.append("".join(s))
    return out


def predict(model, idx, bs=256):
    model.eval(); res = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            res.append(model(prep(X[idx[i:i + bs]])))
    return torch.cat(res)


if __name__ == "__main__":
    model = CRNN()
    print("params", sum(p.numel() for p in model.parameters()))
    EPOCHS, BS = 30, 32
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS * (len(tr) // BS + 1))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    best = 9; log = []
    for ep in range(EPOCHS):
        model.train(); perm = np.random.permutation(len(tr)); t0 = time.time(); tl = 0
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            xb = augment(prep(X[tr[b]]))
            tg = [enc(text[tr[j]]) for j in b]
            lp = model(xb).permute(1, 0, 2)
            loss = ctc(lp, torch.tensor(sum(tg, [])), torch.full((len(b),), lp.shape[0], dtype=torch.long),
                       torch.tensor([len(t) for t in tg]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step(); sched.step()
            tl += loss.item() * len(b)
        pv = greedy(predict(model, va))
        gt = [text[j] for j in va]
        cer = sum(ed(p, g) for p, g in zip(pv, gt)) / sum(len(g) for g in gt)
        em = np.mean([p == g for p, g in zip(pv, gt)])
        log.append(dict(epoch=ep, train_loss=tl / len(tr), val_cer=cer, val_em=float(em)))
        print(f"ep{ep:02d} loss {tl/len(tr):.3f} val CER {cer:.4f} EM {em:.4f} {time.time()-t0:.0f}s", flush=True)
        if cer < best:
            best = cer; torch.save(model.state_dict(), ROOT / "data" / "crnn.pt")
    (ROOT / "data" / "train_log.json").write_text(json.dumps(log, indent=1))
