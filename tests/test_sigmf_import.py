import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from wavelink.sigmf_import import import_sigmf,verify_import
from wavelink.iq_replay import load_recording
from experiments.sigmf_demo import exercise


class SigMFImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.meta=self.root/'capture.sigmf-meta'
        self.data=self.root/'capture.sigmf-data'
        self.pilots=self.root/'pilots.npy'
        np.save(self.pilots,np.ones((1,20,28),complex))
        self.doc={'global':{'core:datatype':'cf32_le','core:sample_rate':1e6,'core:version':'1.2.6'},
                  'captures':[{'core:sample_start':0}],'annotations':[]}
        self.write()
        # Independently packed scalar components establish I/Q order and byte order.
        self.data.write_bytes(struct.pack('<ff',-3.,7.)+struct.pack('<ff',.25,-.5)*1984+struct.pack('<ff',9.,-2.))

    def write(self):self.meta.write_text(json.dumps(self.doc,ensure_ascii=False),encoding='utf-8')

    def convert(self,**kwargs):
        return import_sigmf(self.meta,self.pilots,self.root/'out',slots=1,**kwargs)

    def test_independent_float_bytes_and_complex_sample_offset(self):
        result=self.convert(first_sample=1)
        meta,samples,pilots,available=load_recording(self.root/'out')
        np.testing.assert_array_equal(samples,np.full((1,1984),.25-.5j))
        self.assertEqual((result['prefix_samples'],result['suffix_samples']),(1,1))
        self.assertTrue(available[0])
        self.assertTrue(verify_import(self.meta,self.pilots,self.root/'out')['ok'])
        with self.assertRaises(FileExistsError):self.convert(first_sample=1)

    def test_integer_bytes_require_explicit_scale_and_count_rails(self):
        self.doc['global']['core:datatype']='ci16_le';self.write()
        self.data.write_bytes(struct.pack('<hh',-32768,32767)*1984)
        with self.assertRaises(ValueError):self.convert()
        r=self.convert(integer_scale=1/32768)
        expected=-1+1j*(32767/32768)
        np.testing.assert_array_equal(np.load(self.root/'out/samples.npy'),np.full((1,1984),expected))
        self.assertEqual(r['rail_components'],3968)

    def test_rejects_nonpairs_and_truncated_requested_window(self):
        original=self.data.read_bytes()
        for wire in (original+b'x',original[:100]):
            self.data.write_bytes(wire)
            with self.assertRaises(ValueError):self.convert()
        self.assertFalse((self.root/'out').exists())

    def test_rejects_unsupported_layouts_and_required_extensions(self):
        for key,value in [('core:datatype','cf32_be'),('core:datatype','cu8'),('core:num_channels',2),
                          ('core:offset',4),('core:version','1.3.0'),('core:sample_rate',0),
                          ('core:dataset','other.dat'),('core:metadata_only',True),
                          ('core:extensions',[{'name':'custom','optional':False}])]:
            old=self.doc['global'].copy();self.doc['global'][key]=value;self.write()
            with self.assertRaises(ValueError):self.convert()
            self.doc['global']=old
        for captures in ([{'core:sample_start':0},{'core:sample_start':100}],
                         [{'core:sample_start':1}],[{'core:sample_start':0,'core:header_bytes':0}]):
            self.doc['captures']=captures;self.write()
            with self.assertRaises(ValueError):self.convert()

    def test_sha512_and_availability_validation(self):
        self.doc['global']['core:sha512']='0'*128;self.write()
        with self.assertRaises(ValueError):self.convert()
        self.doc['global']['core:sha512']=hashlib.sha512(self.data.read_bytes()).hexdigest();self.write()
        a=self.root/'available.npy';np.save(a,[1])
        with self.assertRaises(ValueError):self.convert(available=a)
        np.save(a,[False])
        self.convert(available=a)
        self.assertFalse(load_recording(self.root/'out')[3][0])

    def test_annotations_do_not_select_timing_and_empty_capture_is_allowed(self):
        self.doc['captures']=[]
        self.doc['annotations']=[{'core:sample_start':1980,'core:label':'真实起点不得用于选帧'}]
        self.write();r=self.convert()
        self.assertEqual(r['first_sample'],0)
        self.assertEqual(r['ignored_annotations'],1)
        self.assertEqual(np.load(self.root/'out/samples.npy')[0,0],-3+7j)

    def test_input_and_output_tampering_fail_verification(self):
        self.convert()
        original=self.data.read_bytes();self.data.write_bytes(original+b'xx')
        self.assertFalse(verify_import(self.meta,self.pilots,self.root/'out')['ok'])
        self.data.write_bytes(original)
        path=self.root/'out/samples.npy';path.write_bytes(path.read_bytes()+b'changed')
        self.assertFalse(verify_import(self.meta,self.pilots,self.root/'out')['ok'])

    def test_invalid_selection_pilots_and_nonfinite_samples(self):
        for first in (-1,True,1.5):
            with self.assertRaises(ValueError):self.convert(first_sample=first)
        self.data.write_bytes(struct.pack('<ff',float('nan'),0)*1984)
        with self.assertRaises(ValueError):self.convert()
        self.data.write_bytes(struct.pack('<ff',0,0)*1984)
        np.save(self.pilots,np.full((1,20,28),2+0j))
        with self.assertRaises(ValueError):self.convert()

    def test_all_wire_formats_match_direct_quantized_receiver(self):
        rows=exercise(self.root/'integration')
        self.assertEqual(len(rows),3)
        self.assertTrue(all(r['direct_quantized_reference_mismatches']==0 and r['events_equal'] for r in rows))
        self.assertEqual(rows[1]['evaluator_clipped_components'],0)
        self.assertLessEqual(rows[1]['sample_max_error'],np.sqrt(2)*rows[1]['scalar_scale']/2+1e-12)
        self.assertGreater(rows[2]['evaluator_clipped_components'],0)
        self.assertGreater(rows[2]['quantization_nmse'],rows[1]['quantization_nmse'])


if __name__ == '__main__':unittest.main()
