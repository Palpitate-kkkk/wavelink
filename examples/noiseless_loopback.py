"""A sanity check, not an over-the-air performance measurement."""

import numpy as np
from wavelink import ofdm_demodulate, ofdm_modulate, qpsk_demodulate, qpsk_modulate


def main():
    rng = np.random.default_rng(20261028)
    bits = rng.integers(0, 2, 100 * 64 * 2, dtype=np.uint8)
    grid = qpsk_modulate(bits).reshape(100, 64)
    samples = ofdm_modulate(grid, cp_len=16)
    recovered = ofdm_demodulate(samples, n_fft=64, cp_len=16)
    errors = int(np.count_nonzero(bits != qpsk_demodulate(recovered.ravel())))
    print(f"Noiseless simulation | bits={bits.size} | bit_errors={errors} | BER={errors / bits.size:.6g}")
    if errors:
        raise RuntimeError("Noiseless loopback failed")


if __name__ == "__main__":
    main()
