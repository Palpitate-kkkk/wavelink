"""Replay the explicit WaveLink fixed-slot IQ format, without scorer truth.

This is not a continuous-stream detector or a generic SDR file decoder.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from .baseband import ofdm_demodulate, qpsk_demodulate
from .acquisition import acquire
from .alignment import align_marker
from .reference import ReferenceTrackedLMMSE
from .receiver import comb_grid, zero_force
from .synchronization import rotate_frequency


INPUTS = ('metadata.json', 'samples.npy', 'pilots.npy', 'available.npy')
ARTIFACTS = ('frames.csv', 'bits.npy', 'events.json', 'summary.json')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def implementation_hashes():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def load_recording(directory):
    directory = Path(directory)
    meta = json.loads((directory/'metadata.json').read_text())
    fields = {'format', 'sample_rate_hz', 'n_slots', 'slot_samples', 'n_fft', 'cp_len'}
    if not isinstance(meta, dict) or set(meta) != fields:
        raise ValueError('exact v1 metadata fields required; scorer truth belongs in a separate file')
    if meta['format'] != 'wavelink-iq-v1' or meta['slot_samples'] != 1984 or meta['n_fft'] != 64 or meta['cp_len'] != 16:
        raise ValueError('only the documented 1984-sample / 64FFT / 16CP format is supported')
    n, fs = meta['n_slots'], meta['sample_rate_hz']
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('positive integer n_slots required')
    if isinstance(fs, bool) or not isinstance(fs, (int, float)) or not np.isfinite(fs) or fs <= 0:
        raise ValueError('finite positive sample_rate_hz required')
    def array(name):
        value = np.load(directory/name, allow_pickle=False, mmap_mode='r')
        if not isinstance(value, np.ndarray):
            value.close()
            raise ValueError('single .npy arrays required; archives are not supported')
        return value
    samples = array('samples.npy')
    pilots = array('pilots.npy')
    available = array('available.npy')
    if samples.shape != (n, 1984) or samples.dtype.kind != 'c' or samples.dtype.itemsize not in (8, 16):
        raise ValueError('samples must be complex64/128 (n_slots,1984)')
    if pilots.shape != (n, 20, 28) or pilots.dtype.kind != 'c' or pilots.dtype.itemsize not in (8, 16):
        raise ValueError('pilots must be complex64/128 (n_slots,20,28)')
    if available.shape != (n,) or available.dtype != np.dtype(bool):
        raise ValueError('available must be a Boolean vector')
    # Validate the whole recording before decoding or writing any outputs.
    for i in range(n):
        if not np.all(np.isfinite(samples[i])) or not np.all(np.isfinite(pilots[i])) or not np.allclose(abs(pilots[i].astype(complex)), 1., rtol=0, atol=1e-12):
            raise ValueError('finite samples and known unit-modulus pilots required, including missing slots')
    return meta, samples, pilots, available


def receive_recording(directory):
    """Input contains samples/known pilots/availability only; no BER scoring."""
    directory = Path(directory)
    hashes = {name: digest(directory/name) for name in INPUTS}
    implementation = implementation_hashes()
    meta, samples, pilots, available = load_recording(directory)
    layout = comb_grid(2)
    receiver = ReferenceTrackedLMMSE(layout, gate=.25)
    bits = np.full((meta['n_slots'], 960), -1, dtype=np.int8)
    rows = []
    for slot in range(meta['n_slots']):
        final = slot == meta['n_slots']-1
        row = dict(slot=slot, status='missing', estimated_start='', estimated_epsilon='',
                   estimated_cfo_hz='', score='', reference_shift='')
        y = None
        shift = 0
        if available[slot]:
            proposal = acquire(samples[slot], .2)
            row.update(status='rejected', estimated_start=proposal.start, estimated_epsilon=proposal.epsilon,
                       estimated_cfo_hz=proposal.epsilon*meta['sample_rate_hz']/64, score=proposal.score)
            if proposal.accepted:
                shift = align_marker(samples[slot], proposal, 8.).shift
                first = proposal.start+224
                corrected = rotate_frequency(samples[slot, first:first+21*80], -proposal.epsilon, 64, first)
                y = ofdm_demodulate(corrected, 64, 16)[1:]
                row.update(status='decoded', reference_shift=shift)
        h = receiver.step(y, pilots[slot] if y is not None else None, shift=shift, final=final)
        if h is not None:
            bits[slot] = qpsk_demodulate(zero_force(y[:, layout.bins(layout.data_tones)], h).ravel())
        row.update(statistics_consumed=receiver.last['statistics_consumed'],
                   ready_before=receiver.last['ready_before'], updated=receiver.last['updated'],
                   observation_clock=receiver.last['observation_clock'])
        rows.append(row)
    if any(digest(directory/name) != value for name, value in hashes.items()) or implementation_hashes() != implementation:
        raise RuntimeError('input/implementation changed during replay')
    decoded = sum(r['status']=='decoded' for r in rows)
    summary = dict(slots=len(rows), decoded_slots=decoded, rejected_slots=sum(r['status']=='rejected' for r in rows),
                   missing_slots=sum(r['status']=='missing' for r in rows), offered_bits=len(rows)*960,
                   decoded_bits=decoded*960, erased_bits=(len(rows)-decoded)*960,
                   statistics_consumed=receiver.consumed, bootstrap_slot=receiver.bootstrap_slot,
                   final_slot_consumed=rows[-1]['statistics_consumed'],
                   interpretation='Decoded does not certify correct timing, frequency, payload or CRC; no truth enters this receiver and no BER is computed.')
    return dict(rows=rows, bits=bits, events=receiver.events, summary=summary,
                input_sha256=hashes, implementation_sha256=implementation, recording_metadata=meta)


def save_replay(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite a previous run; every comparison has a separate path.
    if any((output/name).exists() for name in (*ARTIFACTS, 'metadata.json')):
        raise ValueError('output artifacts already exist; use a new output directory')
    with (output/'frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=result['rows'][0].keys())
        writer.writeheader()
        writer.writerows(result['rows'])
    np.save(output/'bits.npy', result['bits'], allow_pickle=False)
    (output/'events.json').write_text(json.dumps(result['events'], indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(result['summary'], indent=2)+'\n')
    meta = dict(format='wavelink-iq-output-v1',
                runtime=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()),
                policy=dict(capture_threshold=.2, pilot_gate=.25, marker_multiple=8., physical_window='original acquisition'),
                input_sha256=result['input_sha256'],
                implementation_sha256=result['implementation_sha256'], recording_metadata=result['recording_metadata'],
                artifact_sha256={name: digest(output/name) for name in ARTIFACTS})
    (output/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')


def verify_replay(recording, output):
    """Check the explicit input, current implementation and fixed output files."""
    failures = []
    try:
        meta = json.loads((Path(output)/'metadata.json').read_text())
        if not isinstance(meta, dict) or meta.get('format') != 'wavelink-iq-output-v1':
            raise ValueError('unsupported output manifest')
        for field, base, names in [('input_sha256', Path(recording), INPUTS), ('artifact_sha256', Path(output), ARTIFACTS)]:
            declared = meta.get(field)
            if not isinstance(declared, dict) or set(declared) != set(names):
                failures.append('invalid manifest: '+field)
                continue
            for name in names:
                if digest(base/name) != declared[name]: failures.append('hash mismatch: '+field+'/'+name)
        if meta.get('implementation_sha256') != implementation_hashes():
            failures.append('current implementation differs')
    except (OSError, ValueError, UnicodeError) as exc:
        failures.append(str(exc))
    return dict(ok=not failures, failures=failures,
                scope='Consistency only; manifest is not signed, and this does not establish physical reception or payload correctness.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'verify'])
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == 'run':
            result = receive_recording(args.recording)
            save_replay(result, args.output)
            print(json.dumps(result['summary'], indent=2))
        else:
            result = verify_replay(args.recording, args.output)
            print(json.dumps(result, indent=2))
            if not result['ok']: raise SystemExit(1)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, str(exc)+'\n')


if __name__ == '__main__':
    main()
