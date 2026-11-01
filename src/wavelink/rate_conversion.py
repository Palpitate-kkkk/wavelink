"""Known-rate offline complex resampling; no clock estimator or energy reset."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import numpy as np
from .iq_replay import digest, implementation_hashes


def ratio_from_rates(input_rate, output_rate):
    for rate in (input_rate,output_rate):
        if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or not 0 < rate <= 1e12:
            raise ValueError('finite declared rates in (0,1e12] required')
    ratio=Fraction(str(output_rate))/Fraction(str(input_rate))
    if not Fraction(1,2) <= ratio <= 2 or max(ratio.numerator,ratio.denominator)>10_000_000:
        raise ValueError('supported exact rate ratio is0.5..2 with reduced factors<=10million; no rate approximation')
    return ratio.numerator,ratio.denominator


def _kernel(positions,length,cutoff,radius):
    indices=np.floor(positions).astype(np.int64)[:,None]+np.arange(-radius+1,radius+1)
    distance=positions[:,None]-indices
    window=np.where(abs(distance)<radius,.5+.5*np.cos(np.pi*distance/radius),0.)
    weights=cutoff*np.sinc(cutoff*distance)*window
    weights/=weights.sum(axis=1,keepdims=True)
    # Normalize the full phase kernel, then zero-pad outside the recording.
    weights*=((indices>=0)&(indices<length))
    return np.clip(indices,0,length-1),weights


def resample_complex(samples,up,down,*,output_count=None,radius=32,rolloff=.94,chunk_size=4096):
    x=np.asarray(samples)
    if x.ndim!=1 or not x.size or x.dtype.kind!='c' or x.dtype.itemsize not in (8,16) or not np.all(np.isfinite(x)):
        raise ValueError('finite nonempty complex64/128 vector required')
    for value,name,minimum in [(up,'up',1),(down,'down',1),(radius,'radius',8),(chunk_size,'chunk_size',1)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<minimum:
            raise ValueError(name+' must be an integer >= '+str(minimum))
    if radius>128 or isinstance(rolloff,bool) or not isinstance(rolloff,(int,float)) or not math.isfinite(rolloff) or not .5<=rolloff<=1:
        raise ValueError('radius<=128 and finite rolloff in[.5,1] required')
    divisor=math.gcd(int(up),int(down));up,down=int(up)//divisor,int(down)//divisor
    if max(up,down)>10_000_000 or not .5<=up/down<=2:
        raise ValueError('supported reduced factors<=10million and ratio0.5..2 required')
    natural=(len(x)*up+down-1)//down
    count=natural if output_count is None else output_count
    if isinstance(count,(bool,np.bool_)) or not isinstance(count,(int,np.integer)) or not 1<=count<=natural:
        raise ValueError('output_count must select1..ceil(input_length*up/down); no extra padding')
    count=int(count)
    output=np.empty(count,complex)
    cutoff=rolloff*min(1.,up/down)
    noise_gain=0.
    if up==down:
        output[:]=x[:count]
        noise_gain=float(count)
    else:
        for first in range(0,count,chunk_size):
            last=min(first+chunk_size,count)
            positions=np.arange(first,last,dtype=float)*(down/up)
            indices,weights=_kernel(positions,len(x),cutoff,radius)
            output[first:last]=np.sum(x[indices]*weights,axis=1)
            noise_gain+=float(np.sum(weights**2))
    info=dict(input_samples=len(x),output_samples=count,natural_output_samples=natural,explicitly_trimmed_tail=natural-count,
              up=up,down=down,first_output_input_position=0.,last_output_input_position=(count-1)*down/up,
              input_lookahead_samples=0 if up==down else int(radius),radius=int(radius),rolloff=rolloff,
              cutoff_cycles_per_input_sample=None if up==down else cutoff/2,
              iid_noise_variance_gain_mean=noise_gain/count,
              filter='identity' if up==down else 'DC-normalized Hann-windowed sinc; zero extension',
              scope='Offline known-rate conversion; noncausal neighbor access; no clock estimation, AGC or energy renormalization. Noise becomes colored; boundary samples differ.')
    return output,info


def convert_stream(input_path,output,input_rate,output_rate,*,output_count=None):
    input_path,output=Path(input_path),Path(output)
    initial=digest(input_path);implementation=implementation_hashes()
    loaded=np.load(input_path,allow_pickle=False,mmap_mode='r')
    if not isinstance(loaded,np.ndarray):
        loaded.close();raise ValueError('single .npy vector required')
    up,down=ratio_from_rates(input_rate,output_rate)
    y,info=resample_complex(loaded,up,down,output_count=output_count)
    if digest(input_path)!=initial or implementation_hashes()!=implementation:raise RuntimeError('input/implementation changed')
    output.mkdir(parents=True,exist_ok=False)
    np.save(output/'stream.npy',y,allow_pickle=False)
    meta=dict(input_sha256=initial,implementation_sha256=implementation,output_sha256=digest(output/'stream.npy'),
              input_rate_hz=input_rate,output_rate_hz=output_rate,conversion=info,
              runtime=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()))
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-rate',type=float,required=True)
    parser.add_argument('--output-rate',type=float,required=True)
    parser.add_argument('--output-count',type=int)
    args=parser.parse_args()
    try:
        meta=convert_stream(args.input,args.output,args.input_rate,args.output_rate,output_count=args.output_count)
        print(json.dumps(meta['conversion'],indent=2))
    except (OSError,ValueError,RuntimeError) as exc:parser.exit(1,str(exc)+'\n')


if __name__ == '__main__':main()
