import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from wavelink.iq_replay import load_recording, receive_recording, save_replay, verify_replay, digest
from wavelink.synchronization import rotate_frequency
from experiments.acquisition_ber import packet
from experiments.iq_file_demo import generate, memory_decode


class IQReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.recording = self.root/'recording'
        generate(self.recording, 3)

    def test_noiseless_payload_rejected_and_missing_are_distinct(self):
        powers = json.loads(Path('experiments/followup.json').read_text())['states']['initial']
        x, p, truth, _, _ = packet(361101, 1, powers, 0.)
        np.save(self.recording/'samples.npy', [rotate_frequency(x,1.2,64), np.zeros(1984,complex), x])
        np.save(self.recording/'pilots.npy', [p,p,p])
        np.save(self.recording/'available.npy', [True,True,False])
        result = receive_recording(self.recording)
        self.assertEqual([r['status'] for r in result['rows']], ['decoded','rejected','missing'])
        np.testing.assert_array_equal(result['bits'][0], truth.ravel())
        self.assertTrue(np.all(result['bits'][1:]==-1))
        self.assertFalse(result['summary']['final_slot_consumed'])
        self.assertEqual(result['summary']['erased_bits'], 1920)
        r=result['rows'][0]
        self.assertAlmostEqual(r['estimated_cfo_hz'], r['estimated_epsilon']*1e6/64)
        quantized=np.load(self.recording/'samples.npy').astype(np.complex64)
        np.save(self.recording/'samples.npy',quantized)
        np.testing.assert_array_equal(receive_recording(self.recording)['bits'][0], truth.ravel())

    def test_file_replay_matches_full_legacy_memory_path_and_never_modifies_input(self):
        directory=self.root/'long'
        x,p,a,_=generate(directory)
        before={name:digest(directory/name) for name in ('metadata.json','samples.npy','pilots.npy','available.npy')}
        expected,events=memory_decode(x,p,a)
        actual=receive_recording(directory)
        np.testing.assert_array_equal(actual['bits'], expected)
        self.assertEqual(actual['events'], events)
        self.assertEqual(actual['summary']['missing_slots'], 4)
        self.assertGreaterEqual(actual['summary']['rejected_slots'], 1)
        self.assertIsNotNone(actual['summary']['bootstrap_slot'])
        self.assertFalse(actual['rows'][-1]['statistics_consumed'])
        self.assertEqual(before,{name:digest(directory/name) for name in before})

    def test_output_manifest_detects_input_and_output_tampering(self):
        result=receive_recording(self.recording)
        output=self.root/'out'
        save_replay(result,output)
        self.assertTrue(verify_replay(self.recording,output)['ok'])
        with self.assertRaises(ValueError):save_replay(result,output)
        path=output/'bits.npy';original=path.read_bytes();path.write_bytes(original+b'changed')
        self.assertFalse(verify_replay(self.recording,output)['ok'])
        path.write_bytes(original)
        path=self.recording/'metadata.json';path.write_text(path.read_text()+' ')
        self.assertFalse(verify_replay(self.recording,output)['ok'])

    def test_unknown_truth_fields_are_rejected(self):
        path=self.recording/'metadata.json';meta=json.loads(path.read_text());meta['true_epsilon']=1.2
        path.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):load_recording(self.recording)

    def test_invalid_shapes_types_and_nonfinite_values(self):
        for samples in (np.zeros((3,1983),complex),np.zeros((3,1984)),np.full((3,1984),np.nan+0j)):
            np.save(self.recording/'samples.npy',samples)
            with self.assertRaises(ValueError):load_recording(self.recording)
        with (self.recording/'samples.npy').open('wb') as stream:
            np.savez(stream, samples=np.zeros((3,1984),complex))
        with self.assertRaises(ValueError):load_recording(self.recording)
        np.save(self.recording/'samples.npy',np.zeros((3,1984),np.complex64))
        self.assertEqual(load_recording(self.recording)[1].dtype,np.dtype('complex64'))
        np.save(self.recording/'available.npy',[1,1,1])
        with self.assertRaises(ValueError):load_recording(self.recording)

    def test_invalid_metadata_and_nonunit_pilots(self):
        path=self.recording/'metadata.json';original=path.read_text()
        for field,value in [('sample_rate_hz',0),('n_slots',True),('n_fft',128)]:
            meta=json.loads(original);meta[field]=value;path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):load_recording(self.recording)
        path.write_text(original)
        np.save(self.recording/'pilots.npy',np.full((3,20,28),2+0j))
        with self.assertRaises(ValueError):load_recording(self.recording)


if __name__ == '__main__':
    unittest.main()
