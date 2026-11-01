import copy
import csv
import json
from pathlib import Path
import unittest
from experiments.support_audit import validate, evaluate
from experiments.replay_reference import compare_rows, replay_case
from experiments.reference_ber import validate as validate_reference


class SupportAuditTests(unittest.TestCase):
    def test_protocol_rejects_reused_seeds_and_changed_noise(self):
        cfg = json.loads(Path('experiments/support.json').read_text())
        validate(cfg)
        for key, value in [('noise_variance', .1), ('seeds', list(range(321001, 321017)))]:
            changed = copy.deepcopy(cfg)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate(changed)

    def test_coordinate_control_and_model_support_bias(self):
        cfg = json.loads(Path('experiments/support.json').read_text())
        cfg.update(seeds=[341101], shifts=[0, 12])
        # Summary SE needs two independent episodes.
        cfg['seeds'].append(341102)
        result = evaluate(cfg)
        for row in result['runs']:
            if row['estimator'] == 'transported':
                self.assertLess(row['max_difference_from_raw_ls'], 1e-10)
                self.assertLess(row['data_bias_nmse'], 1e-24)
                self.assertEqual(row['outside_support_power'], 0.)
        causal = [r for r in result['runs'] if r['profile']=='weaker_first' and r['shift']==12 and r['estimator']=='causal']
        self.assertTrue(all(r['data_bias_nmse'] > 1e-3 for r in causal))
        self.assertTrue(all(r['outside_support_power'] > 0 for r in causal))

    def test_saved_low_snr_episode_replays_including_erasure_and_events(self):
        cfg = json.loads(Path('experiments/reference.json').read_text())
        protocol = validate_reference(cfg)
        records = {name: list(csv.DictReader(Path('reports/reference', name+'.csv').read_text().splitlines()))
                   for name in ('runs', 'costs', 'events', 'windows')}
        result = replay_case(cfg, protocol, 'stationary', 6, cfg['seeds'][0], records)
        self.assertEqual(result['matched_rows']['runs'], 16)
        self.assertEqual(result['matched_rows']['costs'], 6)
        self.assertGreater(result['matched_rows']['events'], 0)

    def test_replay_rejects_changed_error_count_mask_and_row_count(self):
        for actual, expected in [([dict(errors=3)], [dict(errors='2')]),
                                 ([dict(mask='abc')], [dict(mask='abd')]),
                                 ([], [dict(errors='0')]),
                                 ([dict(loss=.1)], [dict(loss='.10001')])]:
            with self.assertRaises(AssertionError):
                compare_rows(actual, expected, 'fixture')
        compare_rows([dict(loss=.1)], [dict(loss='.10000000000001')], 'float tolerance')


if __name__ == '__main__':
    unittest.main()
