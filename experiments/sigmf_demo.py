"""Synthetic SigMF wire-format / quantization integration, no hardware data."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import numpy as np
from experiments.iq_file_demo import generate, memory_decode
from experiments.multipath_ber import write_csv
from wavelink.sigmf_import import import_sigmf, verify_import
from wavelink.iq_replay import receive_recording, save_replay, verify_replay, digest


def exercise(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    samples,pilots,available,truth=generate(root/'original')
    x=np.asarray(samples).ravel()
    components=np.column_stack((x.real,x.imag))
    peak=float(np.max(abs(components)))
    baseline=receive_recording(root/'original')
    rows=[]
    for name,datatype,scale in [('float32','cf32_le',None),('int16','ci16_le',peak/30000),('clipped_int16','ci16_le',peak/120000)]:
        directory=root/name
        directory.mkdir()
        clipped=0
        if datatype == 'cf32_le':
            encoded=components.astype('<f4')
            expected=(encoded[:,0].astype(float)+1j*encoded[:,1].astype(float)).reshape(96,1984)
        else:
            codes=np.rint(components/scale)
            clipped=int(np.count_nonzero((codes < -32768)|(codes > 32767)))
            encoded=np.clip(codes,-32768,32767).astype('<i2')
            expected=((encoded[:,0].astype(float)+1j*encoded[:,1].astype(float))*scale).reshape(96,1984)
        # Explicit prefix/suffix tests that selection is in complex samples.
        wire=np.concatenate((np.zeros((7,2),dtype=encoded.dtype),encoded,np.zeros((5,2),dtype=encoded.dtype)))
        data=directory/'capture.sigmf-data'
        wire.tofile(data)
        meta={'global':{'core:datatype':datatype,'core:sample_rate':1e6,'core:version':'1.2.6','core:sha512':hashlib.sha512(data.read_bytes()).hexdigest()},
              'captures':[{'core:sample_start':0}],
              'annotations':[{'core:sample_start':7,'core:label':'ignored diagnostic annotation'}]}
        metadata=directory/'capture.sigmf-meta'
        metadata.write_text(json.dumps(meta,indent=2)+'\n')
        converted=directory/'converted'
        manifest=import_sigmf(metadata,root/'original/pilots.npy',converted,slots=96,first_sample=7,
                              available=root/'original/available.npy',integer_scale=scale)
        np.testing.assert_array_equal(np.load(converted/'samples.npy'),expected)
        assert verify_import(metadata,root/'original/pilots.npy',converted,root/'original/available.npy')['ok']
        actual=receive_recording(converted)
        reference,events=memory_decode(expected,pilots,available)
        np.testing.assert_array_equal(actual['bits'],reference)
        assert actual['events'] == events
        save_replay(actual,directory/'decoded')
        assert verify_replay(converted,directory/'decoded')['ok']
        accepted=actual['bits'] >= 0
        delta=np.asarray(expected).ravel()-x
        rows.append(dict(case=name,datatype=datatype,slots=96,first_sample=7,suffix_samples=manifest['suffix_samples'],
                         scalar_scale=manifest['scalar_scale'],rail_components=manifest['rail_components'],
                         evaluator_clipped_components=clipped,sample_max_error=float(np.max(abs(delta))),
                         quantization_nmse=float(np.sum(abs(delta)**2)/np.sum(abs(x)**2)),
                         decoded_slots=actual['summary']['decoded_slots'],rejected_slots=actual['summary']['rejected_slots'],
                         missing_slots=actual['summary']['missing_slots'],
                         payload_errors=int(np.count_nonzero(actual['bits'][accepted] != truth[accepted])),
                         erased_bits=actual['summary']['erased_bits'],
                         decisions_changed_vs_original=int(np.count_nonzero(actual['bits'] != baseline['bits'])),
                         direct_quantized_reference_mismatches=0,events_equal=True))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    paths=sorted(Path('src').rglob('*.py'))+sorted(Path('experiments').glob('*.py'))+sorted(Path('experiments').glob('*.json'))
    hashes={str(p):digest(p) for p in paths}
    temporary=tempfile.TemporaryDirectory(prefix='wavelink-sigmf-') if args.output is None else None
    output=Path(temporary.name)/'run' if temporary else args.output
    rows=exercise(output)
    if any(digest(p) != h for p,h in hashes.items()): raise RuntimeError('source changed')
    if args.report:
        args.report.mkdir(parents=True,exist_ok=True)
        write_csv(args.report/'runs.csv',rows)
        fingerprints={}
        for row in rows:
            case=output/row['case']
            imported=json.loads((case/'converted/import.json').read_text())
            replay=json.loads((case/'decoded/metadata.json').read_text())
            fingerprints[row['case']]=dict(input_sha256=imported['input_sha256'],converted_sha256=imported['output_sha256'],decoded_sha256=replay['artifact_sha256'])
        meta=dict(source_sha256=hashes,artifact_sha256={'runs.csv':digest(args.report/'runs.csv')},
                  case_fingerprints=fingerprints,
                  base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
                  command=['python','-m','experiments.sigmf_demo',*sys.argv[1:]],
                  runtime=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
                  specification='https://sigmf.org/ (v1.2.6 viewed2026-11-01); reader supports declared1.2.x subset only',
                  scope='Three deterministic encoding/quantization cases on the same96-slot synthetic fixture; not three independent performance trials or hardware measurements. No tuning against this output.')
        (args.report/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
    if temporary: temporary.cleanup()


if __name__ == '__main__':
    main()
