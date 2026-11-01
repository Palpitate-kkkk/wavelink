"""Known-clock integration using independent padded Fourier synthesis.

Not an unknown-clock estimator, analog ADC model, or paired performance trial.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import numpy as np
from wavelink.rate_conversion import resample_complex
from wavelink.iq_replay import receive_recording,digest
from experiments.iq_file_demo import generate
from experiments.multipath_ber import write_csv


def fourier_resize(x,count):
    """Independent periodic DFT resizing, including complex Nyquist splitting."""
    x=np.asarray(x,complex);n=len(x)
    if n==count:return x.copy()
    spectrum=np.fft.fft(x);target=np.zeros(count,complex)
    common=min(n,count);positive=(common+1)//2;negative=(common-1)//2
    target[:positive]=spectrum[:positive]
    if negative:target[-negative:]=spectrum[-negative:]
    if common%2==0:
        if count>n:
            target[common//2]=spectrum[common//2]/2
            target[-common//2]=spectrum[common//2]/2
        else:target[common//2]=spectrum[common//2]+spectrum[-common//2]
    return np.fft.ifft(target)*(count/n)


def synthesize_clock(x,up,down):
    # Guard and total lengths divisible by down make rate and origin exact.
    guard=((256+down-1)//down)*down
    total=((guard+len(x)+guard+down-1)//down)*down
    padded=np.pad(x,(guard,total-guard-len(x)))
    observed=fourier_resize(padded,total*up//down)
    first=guard*up//down;count=(len(x)*up+down-1)//down
    return observed[first:first+count]


def write_recording(path,samples,pilots,available):
    path.mkdir(parents=True,exist_ok=False)
    np.save(path/'samples.npy',samples,allow_pickle=False)
    np.save(path/'pilots.npy',pilots,allow_pickle=False)
    np.save(path/'available.npy',available,allow_pickle=False)
    meta=dict(format='wavelink-iq-v1',sample_rate_hz=1e6,n_slots=len(samples),slot_samples=1984,n_fft=64,cp_len=16)
    (path/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


def evaluate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    samples,pilots,available,truth=generate(output/'original')
    x=np.asarray(samples).ravel();baseline=receive_recording(output/'original')
    rows=[]
    for name,up,down in [('plus100ppm',10001,10000),('minus100ppm',9999,10000),('rate1024k',128,125)]:
        observed=synthesize_clock(x,up,down)
        directory=output/name;directory.mkdir()
        np.save(directory/'observed.npy',observed,allow_pickle=False)
        restored,info=resample_complex(observed,down,up,output_count=len(x))
        if len(restored)!=len(x) or info['explicitly_trimmed_tail'] not in (0,1):
            raise AssertionError('clock conversion changed declared slot count')
        if np.sum(abs(restored-x)**2)/np.sum(abs(x)**2)>.02:
            raise AssertionError('known-rate time-coordinate reconstruction failed fixed fixture')
        raw=np.zeros_like(x);full=min(96,len(observed)//1984)
        raw[:full*1984]=observed[:full*1984]
        raw_available=np.asarray(available).copy();raw_available[full:]=False
        for method,signal,mask in [('uncorrected',raw,raw_available),('known_rate',restored,available)]:
            recording=directory/method
            write_recording(recording,signal.reshape(96,1984),pilots,mask)
            result=receive_recording(recording);accepted=result['bits']>=0
            errors=int(np.count_nonzero(result['bits'][accepted]!=truth[accepted]))
            offered=96*960;erased=result['summary']['erased_bits']
            rows.append(dict(case=name,method=method,declared_input_rate_hz=1e6*up/down,
                             observed_samples=len(observed),full_uncorrected_slots=full,
                             decoded_slots=result['summary']['decoded_slots'],missing_slots=result['summary']['missing_slots'],
                             rejected_slots=result['summary']['rejected_slots'],errors=errors,erased_bits=erased,
                             offered_bits=offered,loss=(errors+erased)/offered,
                             changed_positions_vs_nominal=int(np.count_nonzero(result['bits']!=baseline['bits'])),
                             sample_nmse=float(np.sum(abs(signal-x)**2)/np.sum(abs(x)**2)),
                             filter_lookahead=info['input_lookahead_samples'] if method=='known_rate' else 0,
                             explicit_output_tail_trim=info['explicitly_trimmed_tail'] if method=='known_rate' else 0))
    for row in rows:
        if row['decoded_slots']+row['missing_slots']+row['rejected_slots']!=96:
            raise AssertionError('slot accounting failed')
        if row['erased_bits']!=(96-row['decoded_slots'])*960:
            raise AssertionError('erasure accounting failed')
    rng=np.random.default_rng(371001)
    white=(rng.normal(size=65536)+1j*rng.normal(size=65536))/np.sqrt(2)
    colored,noise_info=resample_complex(white,128,125)
    core=colored[128:-128]
    noise=dict(input_variance=float(np.mean(abs(white)**2)),output_variance=float(np.mean(abs(core)**2)),
               predicted_unit_input_variance=noise_info['iid_noise_variance_gain_mean'],
               lag1_covariance_real=float(np.mean(core[1:]*core[:-1].conj()).real),
               scope='One fixed white-noise realization; expected gain is sum of squared interpolation weights; output is not treated as white AWGN.')
    return rows,noise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    paths=sorted(Path('src').rglob('*.py'))+sorted(Path('experiments').glob('*.py'))+sorted(Path('experiments').glob('*.json'))
    hashes={str(p):digest(p) for p in paths}
    temporary=tempfile.TemporaryDirectory(prefix='wavelink-clock-') if args.output is None else None
    output=Path(temporary.name)/'run' if temporary else args.output
    rows,noise=evaluate(output)
    if any(digest(p)!=h for p,h in hashes.items()):raise RuntimeError('source changed')
    if args.report:
        args.report.mkdir(parents=True,exist_ok=True)
        write_csv(args.report/'runs.csv',rows)
        (args.report/'noise.json').write_text(json.dumps(noise,indent=2)+'\n')
        meta=dict(source_sha256=hashes,artifact_sha256={n:digest(args.report/n) for n in ('runs.csv','noise.json')},
                  base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
                  runtime=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
                  command=['python','-m','experiments.clock_audit',*sys.argv[1:]],
                  protocol='Fixed radius32/rolloff.94, known rates and output count; Fourier padded synthesis independent of sinc correction; same96-slot fixture reused; all slots including bootstrap scored.',
                  scope='Deterministic integration and numerical audit, no new independent BER improvement claim. Centered offline filtering uses up to32 future input samples, changes boundary/noise and prior acquisition information budget. No unknown-clock estimation, causal real-time or hardware claim.')
        (args.report/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(runs=rows,noise=noise),indent=2))
    if temporary:temporary.cleanup()


if __name__=='__main__':main()
