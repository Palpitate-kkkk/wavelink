"""Unit-energy QPSK and unitary OFDM, using native NumPy FFT bin order."""

import numpy as np


def qpsk_modulate(bits):
    """Map pairs (b0,b1) to ((1-2*b0) + j*(1-2*b1))/sqrt(2)."""
    bits = np.asarray(bits)
    if bits.ndim != 1 or bits.size == 0 or bits.size % 2:
        raise ValueError("bits must be a nonempty, even-length vector")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only 0 and 1")
    pairs = bits.astype(np.float64).reshape(-1, 2)
    return ((1 - 2 * pairs[:, 0]) + 1j * (1 - 2 * pairs[:, 1])) / np.sqrt(2)


def qpsk_demodulate(symbols):
    """Hard decisions with bit order matching qpsk_modulate."""
    symbols = np.asarray(symbols, dtype=np.complex128)
    if symbols.ndim != 1 or not symbols.size or not np.all(np.isfinite(symbols)):
        raise ValueError("symbols must be a nonempty finite vector")
    return np.column_stack((symbols.real < 0, symbols.imag < 0)).astype(np.uint8).ravel()


def _validate_sizes(n_fft, cp_len):
    if isinstance(n_fft, bool) or not isinstance(n_fft, (int, np.integer)) or n_fft < 1:
        raise ValueError("n_fft must be a positive integer")
    if isinstance(cp_len, bool) or not isinstance(cp_len, (int, np.integer)):
        raise ValueError("cp_len must be an integer")
    if not 0 <= cp_len <= n_fft:
        raise ValueError("cp_len must be between 0 and n_fft")


def ofdm_modulate(grid, cp_len):
    """Convert a (symbols, subcarriers) grid into serialized CP-OFDM samples."""
    grid = np.asarray(grid, dtype=np.complex128)
    if grid.ndim != 2 or 0 in grid.shape or not np.all(np.isfinite(grid)):
        raise ValueError("grid must be a nonempty finite 2-D array")
    _validate_sizes(grid.shape[1], cp_len)
    useful = np.fft.ifft(grid, axis=-1, norm="ortho")
    blocks = np.concatenate((useful[:, -cp_len:], useful), axis=1) if cp_len else useful
    return blocks.ravel()


def ofdm_demodulate(samples, n_fft, cp_len):
    """Remove CP and FFT complete blocks; assumes perfect symbol timing."""
    _validate_sizes(n_fft, cp_len)
    samples = np.asarray(samples, dtype=np.complex128)
    block_len = n_fft + cp_len
    if samples.ndim != 1 or not samples.size or samples.size % block_len:
        raise ValueError("samples must contain a nonempty whole number of OFDM blocks")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must be finite")
    blocks = samples.reshape(-1, block_len)
    return np.fft.fft(blocks[:, cp_len:], axis=-1, norm="ortho")
