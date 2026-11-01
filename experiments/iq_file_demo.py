"""Generate synthetic IQ files and verify disk replay against an in-memory path."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import platform
import tempfile
import sys
import numpy as np
from wavelink.acquisition import acquire
from wavelink.alignment import align_marker
from wavelink.reference import ReferenceTrackedLMMSE
from wavelink.receiver import comb_grid, pilot_noise_variance, zero_force
from wavelink import qpsk_demodulate
from wavelink.synchronization import rotate_frequency
from wavelink.iq_replay import receive_recording, save_replay, verify_replay, digest
from experiments.acquisition_ber import packet
from experiments.captured_tracking_ber import grid
from experiments.multipath_ber import write_csv


def generate(output, count=96):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(Path('experiments/followup.json').read_text())
    samples, pilots, bits = [], [], []
    available = np.ones(count, dtype=bool)
    for i in range(count):
        powers = protocol['states']['initial'] if i < 80 else protocol['states']['weak15']
        x, p, b, _, _ = packet(361001*100000+i, 1, powers, pilot_noise_variance(18, comb_grid(2), 16))
        observed = rotate_frequency(x, 5.2 if 72 <= i < 80 else 1.2, 64)
        if 64 <= i < 68:
            available[i] = False
            observed = np.zeros_like(observed)
        if i == 68: observed = np.zeros_like(observed)  # available but acquisition rejects
        samples.append(observed)
        pilots.append(p)
        bits.append(b.ravel())
    np.save(output/'samples.npy', samples, allow_pickle=False)
    np.save(output/'pilots.npy', pilots, allow_pickle=False)
    np.save(output/'available.npy', available, allow_pickle=False)
    meta = dict(format='wavelink-iq-v1', sample_rate_hz=1e6, n_slots=count, slot_samples=1984, n_fft=64, cp_len=16)
    (output/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    # Truth remains in the evaluator; no payload bits/start/CFO/taps in recording.
    return samples, pilots, available, np.asarray(bits)


def memory_decode(samples, pilots, available):
    receiver = ReferenceTrackedLMMSE(comb_grid(2), gate=.25)
    decoded = np.full((len(samples), 960), -1, dtype=np.int8)
    for i, x in enumerate(samples):
        y, shift = None, 0
        if available[i]:
            p = acquire(x, .2)
            if p.accepted:
                shift = align_marker(x, p, 8.).shift
                y = grid(x, p.start, p.epsilon)
        h = receiver.step(y, pilots[i] if y is not None else None, shift=shift, final=i == len(samples)-1)
        if h is not None:
            decoded[i] = qpsk_demodulate(zero_force(y[:, receiver.layout.bins(receiver.layout.data_tones)], h).ravel())
    return decoded, receiver.events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='new directory; omit for temporary integration check')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    sources = sorted(Path('src').rglob('*.py'))+sorted(Path('experiments').glob('*.py'))+sorted(Path('experiments').glob('*.json'))
    hashes = {str(p): digest(p) for p in sources}
    temporary = tempfile.TemporaryDirectory(prefix='wavelink-iq-') if args.output is None else None
    output = Path(temporary.name)/'run' if temporary else args.output
    output.mkdir(parents=True, exist_ok=False)
    samples, pilots, available, truth = generate(output/'recording')
    expected, events = memory_decode(samples, pilots, available)
    actual = receive_recording(output/'recording')
    np.testing.assert_array_equal(expected, actual['bits'])
    assert events == actual['events']
    save_replay(actual, output/'decoded')
    assert verify_replay(output/'recording', output/'decoded')['ok']
    accepted = actual['bits'] >= 0
    errors = int(np.count_nonzero(actual['bits'][accepted] != truth[accepted]))
    row = dict(**{k:v for k,v in actual['summary'].items() if k != 'interpretation'},
               payload_errors=errors, in_memory_bit_mismatches=0, events_equal=True,
               input_source='synthetic fixed-slot files, NOT hardware capture')
    if any(digest(p) != h for p,h in hashes.items()): raise RuntimeError('source changed')
    if args.report:
        args.report.mkdir(parents=True, exist_ok=True)
        write_csv(args.report/'runs.csv', [row])
        meta = dict(source_sha256=hashes, artifact_sha256={'runs.csv':digest(args.report/'runs.csv')},
                    base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
                    command=['python','-m','experiments.iq_file_demo',*sys.argv[1:]],
                    runtime=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
                    input_sha256=actual['input_sha256'], output_sha256=json.loads((args.output/'decoded/metadata.json').read_text())['artifact_sha256'],
                    scope='One96-slot synthetic integration fixture; disk replay vs frozen in-memory grid helper. No new BER performance study, continuous-stream or hardware claim.')
        (args.report/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(row, indent=2))
    if temporary: temporary.cleanup()


if __name__ == '__main__':
    main()
