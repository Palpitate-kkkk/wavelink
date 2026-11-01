import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from wavelink.rate_conversion import resample_complex, ratio_from_rates, convert_stream
from experiments.clock_audit import fourier_resize, synthesize_clock


class RateConversionTests(unittest.TestCase):
    def test_identity_is_exact_and_independent_copy(self):
        x=np.array([1+2j,3-4j],np.complex64)
        y,info=resample_complex(x,7,7)
        np.testing.assert_array_equal(x,y)
        y[0]=0
        self.assertEqual(x[0],1+2j)
        self.assertEqual(info['input_lookahead_samples'],0)
        self.assertEqual(info['iid_noise_variance_gain_mean'],1)

    def test_exact_rates_length_and_explicit_trim(self):
        self.assertEqual(ratio_from_rates(1000100,1e6),(10000,10001))
        x=np.ones(101,complex)
        y,info=resample_complex(x,128,125)
        self.assertEqual(len(y),104)
        z,trim=resample_complex(x,128,125,output_count=100)
        np.testing.assert_array_equal(y[:100],z)
        self.assertEqual(trim['explicitly_trimmed_tail'],4)
        self.assertEqual(info['last_output_input_position'],103*125/128)

    def test_analytic_tone_preserves_physical_time_and_amplitude(self):
        n=4096;f=.17
        x=np.exp(2j*np.pi*f*np.arange(n))
        y,_=resample_complex(x,128,125)
        expected=np.exp(2j*np.pi*f*np.arange(len(y))*125/128)
        self.assertLess(np.max(abs(y[64:-64]-expected[64:-64])),.001)

    def test_alias_rejection_and_zero_extension_boundary(self):
        # Input tone .48 cycles/sample exceeds output Nyquist .4 for4/5.
        x=np.exp(2j*np.pi*.48*np.arange(4096))
        y,_=resample_complex(x,4,5)
        self.assertLess(np.max(abs(y[64:-64])),.002)
        dc,_=resample_complex(np.ones(256,complex),128,125)
        np.testing.assert_allclose(dc[64:-64],1,atol=1e-14)
        self.assertGreater(abs(dc[0]-1),.01)

    def test_impulse_origin_and_chunk_invariance(self):
        x=np.zeros(256,complex);x[80]=1
        a,info=resample_complex(x,2,1,chunk_size=7)
        b,_=resample_complex(x,2,1,chunk_size=4096)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(np.argmax(abs(a)),160)
        self.assertEqual(info['first_output_input_position'],0)
        self.assertEqual(info['input_lookahead_samples'],32)

    def test_white_noise_gain_without_power_reset(self):
        rng=np.random.default_rng(371001)
        x=(rng.normal(size=65536)+1j*rng.normal(size=65536))/np.sqrt(2)
        y,info=resample_complex(x,128,125)
        variance=np.mean(abs(y[128:-128])**2)
        self.assertAlmostEqual(variance,info['iid_noise_variance_gain_mean'],delta=.015)
        self.assertGreater(abs(np.mean(y[129:-128]*y[128:-129].conj())),.05)
        self.assertLess(variance,.97)

    def test_fourier_synthesis_independent_tone_and_nyquist(self):
        for n,count in [(31,47),(32,64),(64,32),(47,31)]:
            x=np.exp(2j*np.pi*3*np.arange(n)/n)
            y=fourier_resize(x,count)
            np.testing.assert_allclose(y,np.exp(2j*np.pi*3*np.arange(count)/count),atol=1e-13)
        # Even-length Nyquist bin splits into positive/negative frequencies.
        np.testing.assert_allclose(fourier_resize((-1.)**np.arange(32),64),np.cos(np.pi*np.arange(64)/2),atol=1e-13)
        np.testing.assert_allclose(fourier_resize(np.cos(np.pi*np.arange(64)/2),32),(-1.)**np.arange(32),atol=1e-13)
        x=np.arange(128,dtype=float).astype(complex)
        np.testing.assert_allclose(synthesize_clock(x,1,1),x,atol=1e-12)
        self.assertEqual(len(synthesize_clock(x,10001,10000)),129)

    def test_rejects_invalid_or_implicitly_approximated_rates_and_samples(self):
        for rate in (0,-1,True,float('nan'),float('inf'),1e13):
            with self.assertRaises(ValueError):ratio_from_rates(rate,1e6)
        with self.assertRaises(ValueError):ratio_from_rates(1e6,1000000.001)
        for x in ([1,2],np.ones((2,2),complex),np.array([complex('nan')]),np.array([],complex)):
            with self.assertRaises(ValueError):resample_complex(x,1,1)
        for kwargs in ({'up':True},{'up':3},{'down':0},{'radius':7},{'rolloff':float('nan')},{'output_count':5},{'output_count':False}):
            args=dict(up=1,down=1);args.update(kwargs)
            with self.assertRaises(ValueError):resample_complex(np.ones(4,complex),**args)

    def test_stream_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);np.save(root/'in.npy',np.ones(128,complex))
            result=convert_stream(root/'in.npy',root/'out',1e6,1024000)
            saved=json.loads((root/'out/metadata.json').read_text())
            self.assertEqual(saved,result)
            self.assertEqual(np.load(root/'out/stream.npy').shape,(132,))
            self.assertEqual(len(saved['output_sha256']),64)
            with self.assertRaises(FileExistsError):convert_stream(root/'in.npy',root/'out',1e6,1024000)
