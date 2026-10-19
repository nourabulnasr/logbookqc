"""Small CRNN+CTC handwriting recogniser for numeric log cells, plus CTC prefix beam search (n-best)."""
import math, numpy as np, torch, torch.nn as nn

ALPHA = "0123456789.-"          # CTC class 0 is blank, class i+1 = ALPHA[i]
H, MAXW = 32, 96


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o, pool):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(pool)]
        self.cnn = nn.Sequential(*blk(1, 16, (2, 2)), *blk(16, 32, (2, 2)), *blk(32, 64, (2, 1)), *blk(64, 128, (2, 1)))
        self.rnn = nn.GRU(128 * 2, 96, num_layers=1, bidirectional=True, batch_first=True)
        self.out = nn.Linear(192, len(ALPHA) + 1)

    def forward(self, x):                       # x: (B,1,32,96) in [0,1], ink = high
        f = self.cnn(x)                         # (B,128,2,24)
        b, c, h, t = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, t, c * h)
        y, _ = self.rnn(f)
        return self.out(y).log_softmax(-1)      # (B,T,C)


def prep(X):
    """uint8 (N,32,96) paper-white=255 -> float ink-high tensor (N,1,32,160)."""
    return torch.from_numpy(1.0 - X.astype(np.float32) / 255.0).unsqueeze(1)


def _lse(a, b):
    if a == -math.inf: return b
    if b == -math.inf: return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_search(logp, beam=24, topk=6, nbest=8):
    """CTC prefix beam search. logp: (T,C) numpy log-probs. Returns [(string, logprob)] best-first."""
    beams = {(): (0.0, -math.inf)}            # prefix -> (log p_blank_end, log p_nonblank_end)
    for t in range(logp.shape[0]):
        row = logp[t]
        cand = np.argsort(row)[-topk:]
        nxt = {}
        def add(p, pb, pnb):
            ob, onb = nxt.get(p, (-math.inf, -math.inf))
            nxt[p] = (_lse(ob, pb), _lse(onb, pnb))
        for pre, (pb, pnb) in beams.items():
            tot = _lse(pb, pnb)
            for c in cand:
                lp = float(row[c])
                if c == 0:
                    add(pre, tot + lp, -math.inf)
                    continue
                npre = pre + (int(c),)
                if pre and pre[-1] == c:
                    add(npre, -math.inf, pb + lp)       # repeat only after a blank
                    add(pre, -math.inf, pnb + lp)       # collapse
                else:
                    add(npre, -math.inf, tot + lp)
        beams = dict(sorted(nxt.items(), key=lambda kv: -_lse(*kv[1]))[:beam])
    res = [("".join(ALPHA[i - 1] for i in p), _lse(*s)) for p, s in beams.items()]
    res.sort(key=lambda r: -r[1])
    return res[:nbest]
