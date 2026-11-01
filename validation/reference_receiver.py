"""Independent small receiver reference: Python standard library only.

No NumPy, WaveLink imports, FFT library, pseudoinverse or production helpers.
Dense DFT and pivoted elimination are intentionally small validation tools.
"""
import cmath
from functools import lru_cache
import math
import random


@lru_cache(maxsize=None)
def basis(n, inverse=False):
    sign = 1 if inverse else -1
    return tuple(tuple(cmath.exp(sign*2j*math.pi*k*t/n)/math.sqrt(n) for t in range(n)) for k in range(n))


def transform(x, inverse=False):
    return [sum(a*b for a, b in zip(row, x)) for row in basis(len(x), inverse)]


def solve(a, b):
    """Solve complex A X=B using row pivoting; B contains multiple columns."""
    n = len(a)
    if not n or any(len(row) != n for row in a) or len(b) != n:
        raise ValueError('square system and matching RHS required')
    width = len(b[0])
    if not width or any(len(row) != width for row in b):
        raise ValueError('nonempty rectangular RHS required')
    rows = [[complex(v) for v in ar]+[complex(v) for v in br] for ar, br in zip(a, b)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(rows[i][k]))
        if abs(rows[pivot][k]) < 1e-14:
            raise ValueError('singular reference system')
        rows[k], rows[pivot] = rows[pivot], rows[k]
        scale = rows[k][k]
        rows[k] = [v/scale for v in rows[k]]
        for i in range(n):
            if i != k:
                factor = rows[i][k]
                rows[i] = [v-factor*w for v, w in zip(rows[i], rows[k])]
    return [row[n:] for row in rows]


def estimate(grids, pilots, pilot_tones, data_tones, powers, variance, lmmse):
    fp = [[cmath.exp(-2j*math.pi*k*d/64) for d in range(len(powers))] for k in pilot_tones]
    fd = [[cmath.exp(-2j*math.pi*k*d/64) for d in range(len(powers))] for k in data_tones]
    z = [[g[k % 64]/p[j] for g, p in zip(grids, pilots)] for j, k in enumerate(pilot_tones)]
    if lmmse:
        cov = [[sum(fp[i][d]*powers[d]*fp[j][d].conjugate() for d in range(len(powers)))
                + (variance if i == j else 0) for j in range(len(fp))] for i in range(len(fp))]
        weights = solve(cov, z)
        cross = [[sum(row[d]*powers[d]*fp[j][d].conjugate() for d in range(len(powers)))
                  for j in range(len(fp))] for row in fd]
        hd = [[sum(row[j]*weights[j][t] for j in range(len(fp))) for t in range(len(grids))] for row in cross]
    else:
        gram = [[sum(row[i].conjugate()*row[j] for row in fp) for j in range(len(powers))] for i in range(len(powers))]
        rhs = [[sum(fp[j][d].conjugate()*z[j][t] for j in range(len(fp))) for t in range(len(grids))] for d in range(len(powers))]
        taps = solve(gram, rhs)
        hd = [[sum(row[d]*taps[d][t] for d in range(len(powers))) for t in range(len(grids))] for row in fd]
    return [list(row) for row in zip(*hd)]


def fixture(seed, variance):
    rng = random.Random(seed)
    active = list(range(-26, 0))+list(range(1, 27))
    pilot_tones = sorted(set(list(range(-26, 0, 2))+[-1]+list(range(1, 27, 2))+[26]))
    data_tones = [k for k in active if k not in pilot_tones]
    symbol = lambda: complex(1-2*rng.randrange(2), 1-2*rng.randrange(2))/math.sqrt(2)
    data = [[symbol() for _ in data_tones] for _ in range(21)]
    pilots = [[symbol() for _ in pilot_tones] for _ in range(21)]
    grids = []
    for ds, ps in zip(data, pilots):
        g = [0j]*64
        for k, v in zip(data_tones, ds): g[k % 64] = v
        for k, v in zip(pilot_tones, ps): g[k % 64] = v
        grids.append(g)
    useful = [transform(g, True) for g in grids]
    tx = [v for block in useful for v in block[-16:]+block]
    powers = [.8]+[0.]*3+[.12]+[0.]*7+[.08]
    taps = [complex(rng.gauss(0, 1), rng.gauss(0, 1))*math.sqrt(p/2) for p in powers]
    latent = [sum(taps[d]*tx[t-d] for d in range(min(t+1, len(taps)))) for t in range(len(tx))]
    noise = [complex(rng.gauss(0, 1), rng.gauss(0, 1))*math.sqrt(variance/2) for _ in tx]
    rx = [a+b for a, b in zip(latent, noise)]
    received = [transform(rx[t+16:t+80]) for t in range(0, len(rx), 80)][1:]
    ps = pilots[1:]
    ls = estimate(received, ps, pilot_tones, data_tones, powers, variance, False)
    # A strictly positive assumed noise regularizes LMMSE even in noiseless data.
    mmse = estimate(received, ps, pilot_tones, data_tones, powers, max(variance, .05), True)
    bits = [bit for row in data[1:] for v in row for bit in (int(v.real < 0), int(v.imag < 0))]
    def decode(h):
        return [bit for g, row in zip(received, h) for k, v in zip(data_tones, row)
                for bit in (int((g[k % 64]/v).real < 0), int((g[k % 64]/v).imag < 0))]
    return dict(grids=grids, pilots=pilots, tx=tx, taps=taps, powers=powers, latent=latent, rx=rx,
                received=received, ls=ls, lmmse=mmse, bits=bits, ls_bits=decode(ls), lmmse_bits=decode(mmse))
