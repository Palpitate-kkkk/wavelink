import unittest
from validation.reference_receiver import solve, transform
from experiments.crosscheck import evaluate


class IndependentReferenceTests(unittest.TestCase):
    def test_dense_transform_known_impulse_and_inverse(self):
        self.assertEqual(transform([1, 0, 0, 0]), [.5+0j]*4)
        x = [1+2j, -3j, 4, -.2]
        for a, b in zip(transform(transform(x), True), x):
            self.assertAlmostEqual(abs(a-b), 0., places=12)

    def test_pivoting_solver_and_singular_rejection(self):
        solution = solve([[0, 2], [1, 3]], [[4], [7]])
        self.assertAlmostEqual(abs(solution[0][0]-1), 0.)
        self.assertAlmostEqual(abs(solution[1][0]-2), 0.)
        with self.assertRaises(ValueError): solve([[1, 1], [2, 2]], [[1], [2]])

    def test_full_chain_uses_independent_reference_and_noiseless_decode(self):
        rows = evaluate(dict(seeds=[351101], noise_variances=[0., .05], absolute_tolerance=2e-10))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['ls_bit_errors'], 0)
        self.assertTrue(all(r['ls_decision_mismatches']==r['lmmse_decision_mismatches']==0 for r in rows))


if __name__ == '__main__':
    unittest.main()
