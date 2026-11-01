"""Restricted SigMF1.2.x import to explicit WaveLink fixed slots.

Supports single-channel cf32_le/ci16_le, one contiguous capture at index0.
Not a general schema validator, resampler, annotation-driven packet detector,
or hardware-calibration tool. Integer scale is a user choice, not SigMF gain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import numpy as np
from .iq_replay import digest, implementation_hashes


RECORDING_FILES = ('metadata.json', 'samples.npy', 'pilots.npy', 'available.npy')


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name+' must be an integer >= '+str(minimum))
    return value


def _paths(metadata, pilots, available):
    metadata = Path(metadata)
    if metadata.suffix != '.sigmf-meta':
        raise ValueError('a .sigmf-meta path is required')
    paths = dict(metadata=metadata, data=metadata.with_suffix('.sigmf-data'), pilots=Path(pilots))
    if available is not None: paths['available'] = Path(available)
    return paths


def _array(path):
    result = np.load(path, allow_pickle=False)
    if not isinstance(result, np.ndarray):
        result.close()
        raise ValueError('single .npy array required')
    return result


def import_sigmf(metadata, pilots, output, *, slots, first_sample=0, available=None, integer_scale=None):
    """No annotation, true timing, channel, CFO or payload enters selection."""
    slots = _integer(slots, 'slots', 1)
    first_sample = _integer(first_sample, 'first_sample')
    paths = _paths(metadata, pilots, available)
    hashes = {role: digest(path) for role,path in paths.items()}
    implementation = implementation_hashes()
    meta = json.loads(paths['metadata'].read_text(encoding='utf-8'))
    if not isinstance(meta, dict) or not all(k in meta for k in ('global','captures','annotations')):
        raise ValueError('SigMF global/captures/annotations are required')
    g, captures, annotations = meta['global'], meta['captures'], meta['annotations']
    if not isinstance(g, dict) or not isinstance(captures, list) or not isinstance(annotations, list):
        raise ValueError('invalid SigMF object/array types')
    if not isinstance(g.get('core:version'),str) or not re.fullmatch(r'1\.2\.\d+', g['core:version']):
        raise ValueError('this restricted reader supports SigMF1.2.x only')
    datatype = g.get('core:datatype')
    if datatype not in ('cf32_le','ci16_le'):
        raise ValueError('only cf32_le and ci16_le are supported')
    fs = g.get('core:sample_rate')
    if isinstance(fs,bool) or not isinstance(fs,(float,int)) or not np.isfinite(fs) or not 0 < fs <= 1e12:
        raise ValueError('explicit finite sample rate in (0,1e12] required')
    if _integer(g.get('core:num_channels',1),'core:num_channels',1) != 1 or _integer(g.get('core:offset',0),'core:offset') != 0:
        raise ValueError('only one channel and zero dataset index offset are supported')
    if g.get('core:metadata_only',False) is not False or any(k in g for k in ('core:dataset','core:trailing_bytes')):
        raise ValueError('metadata-only/nonconforming datasets are unsupported')
    extensions = g.get('core:extensions',[])
    if not isinstance(extensions,list) or any(not isinstance(e,dict) or e.get('optional') is not True for e in extensions):
        raise ValueError('required or malformed extensions are unsupported')
    if len(captures) > 1:
        raise ValueError('multiple capture segments are unsupported')
    if captures:
        c=captures[0]
        if not isinstance(c,dict) or 'core:sample_start' not in c:
            raise ValueError('capture sample_start required')
        if _integer(c['core:sample_start'],'core:sample_start') != 0 or _integer(c.get('core:global_index',0),'core:global_index') != 0 or 'core:header_bytes' in c:
            raise ValueError('capture offsets, discontinuities and headers are unsupported')
    if 'core:sha512' in g:
        expected = g['core:sha512']
        if not isinstance(expected,str) or not re.fullmatch('[0-9a-fA-F]{128}',expected):
            raise ValueError('invalid core:sha512')
        h=hashlib.sha512()
        with paths['data'].open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
        if h.hexdigest() != expected.lower(): raise ValueError('SigMF data SHA512 mismatch')
    if datatype == 'ci16_le':
        if isinstance(integer_scale,bool) or not isinstance(integer_scale,(int,float)) or not np.isfinite(integer_scale) or integer_scale <= 0:
            raise ValueError('ci16_le requires an explicit finite positive integer_scale')
        scale=float(integer_scale)
    else:
        if integer_scale is not None: raise ValueError('integer_scale does not apply to cf32_le')
        scale=1.
    dtype=np.dtype('<f4' if datatype == 'cf32_le' else '<i2')
    byte_count=paths['data'].stat().st_size
    if not byte_count or byte_count % (2*dtype.itemsize):
        raise ValueError('data length must contain whole nonempty I/Q pairs')
    total=byte_count//(2*dtype.itemsize)
    count=slots*1984
    if first_sample+count > total:
        raise ValueError('requested full slots exceed the dataset; no truncation is performed')
    p=_array(paths['pilots'])
    a=_array(paths['available']) if available is not None else np.ones(slots,dtype=bool)
    if p.shape != (slots,20,28) or p.dtype.kind != 'c' or p.dtype.itemsize not in (8,16) or not np.all(np.isfinite(p)) or not np.allclose(abs(p.astype(complex)),1.,rtol=0,atol=1e-12):
        raise ValueError('matching known unit-modulus pilots required')
    if a.shape != (slots,) or a.dtype != np.dtype(bool):
        raise ValueError('matching Boolean availability required')
    raw=np.memmap(paths['data'],dtype=dtype,mode='r').reshape(-1,2)
    selected=np.asarray(raw[first_sample:first_sample+count],dtype=np.float64)
    if not np.all(np.isfinite(selected)): raise ValueError('selected floating samples are nonfinite')
    rail=int(np.count_nonzero((selected == -32768)|(selected == 32767))) if datatype == 'ci16_le' else 0
    samples=(selected[:,0]+1j*selected[:,1])*scale
    if not np.all(np.isfinite(samples)): raise ValueError('scaling overflow')
    if any(digest(path) != hashes[role] for role,path in paths.items()) or implementation_hashes() != implementation:
        raise RuntimeError('input/implementation changed during import')
    output=Path(output)
    # Validate first; a fresh destination avoids mixing old/new recordings.
    output.mkdir(parents=True,exist_ok=False)
    np.save(output/'samples.npy',samples.reshape(slots,1984),allow_pickle=False)
    np.save(output/'pilots.npy',p,allow_pickle=False)
    np.save(output/'available.npy',a,allow_pickle=False)
    recording=dict(format='wavelink-iq-v1',sample_rate_hz=fs,n_slots=slots,slot_samples=1984,n_fft=64,cp_len=16)
    (output/'metadata.json').write_text(json.dumps(recording,indent=2)+'\n')
    manifest=dict(format='wavelink-sigmf-import-v1',input_sha256=hashes,implementation_sha256=implementation,
                  output_sha256={name:digest(output/name) for name in RECORDING_FILES},
                  runtime=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
                  source_version=g['core:version'],datatype=datatype,first_sample=first_sample,slots=slots,
                  scalar_scale=scale,prefix_samples=first_sample,suffix_samples=total-first_sample-count,
                  rail_components=rail,ignored_annotations=len(annotations),
                  scope='Explicit fixed-slot selection only. No resampling, gain inference, annotation-driven timing, clipping certification or hardware claim.')
    (output/'import.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def verify_import(metadata,pilots,output,available=None):
    failures=[]
    try:
        manifest=json.loads((Path(output)/'import.json').read_text(encoding='utf-8'))
        if not isinstance(manifest,dict) or manifest.get('format') != 'wavelink-sigmf-import-v1':
            raise ValueError('invalid import manifest')
        actual={role:digest(path) for role,path in _paths(metadata,pilots,available).items()}
        if manifest.get('input_sha256') != actual: failures.append('input fingerprints differ')
        if manifest.get('output_sha256') != {name:digest(Path(output)/name) for name in RECORDING_FILES}:
            failures.append('converted recording differs')
        if manifest.get('implementation_sha256') != implementation_hashes(): failures.append('current implementation differs')
    except (OSError,ValueError,UnicodeError) as exc: failures.append(str(exc))
    return dict(ok=not failures,failures=failures,scope='Unsigned consistency check; no physical or payload authenticity guarantee.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['convert','verify'])
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--pilots',type=Path,required=True)
    parser.add_argument('--available',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--slots',type=int)
    parser.add_argument('--first-sample',type=int,default=0)
    parser.add_argument('--integer-scale',type=float)
    args=parser.parse_args()
    try:
        if args.action == 'convert':
            if args.slots is None: raise ValueError('--slots is required for conversion')
            result=import_sigmf(args.metadata,args.pilots,args.output,slots=args.slots,first_sample=args.first_sample,
                                 available=args.available,integer_scale=args.integer_scale)
            print(json.dumps({k:result[k] for k in ('datatype','slots','first_sample','prefix_samples','suffix_samples','scalar_scale','rail_components')},indent=2))
        else:
            result=verify_import(args.metadata,args.pilots,args.output,args.available)
            print(json.dumps(result,indent=2))
            if not result['ok']: raise SystemExit(1)
    except (OSError,ValueError,RuntimeError) as exc: parser.exit(1,str(exc)+'\n')


if __name__ == '__main__':
    main()
