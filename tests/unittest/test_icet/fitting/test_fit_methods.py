import numpy as np
import unittest

from icet.fitting import fit, available_fit_methods


class TestFitMethods(unittest.TestCase):
    """Unittest class for fit_methods module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        np.random.seed(42)

        # set up dummy linear problem data
        N, M = 200, 50
        self.n_rows, self.n_cols = N, M
        self.A = np.random.normal(0, 1.0, (N, M))
        self.x = np.random.normal(0.0, 1.0, M)
        noise = np.random.normal(0.0, 0.2, N)
        self.y = np.dot(self.A, self.x) + noise

    def shortDescription(self):
        """Prevents unittest from printing docstring in test cases."""
        return None

    def test_all_available_fit_methods(self):
        """Tests all available fit_methods."""

        # without standardize
        for fit_method in available_fit_methods:
            res = fit(self.A, self.y, fit_method=fit_method, standardize=False)
            self.assertLess(np.linalg.norm(self.x - res['parameters']), 0.2)

        # with standardize
        for fit_method in available_fit_methods:
            res = fit(self.A, self.y, fit_method=fit_method, standardize=True)
            self.assertLess(np.linalg.norm(self.x - res['parameters']), 0.2)

    def test_with_large_condition_number(self):
        """Test fit with very large condition number"""

        # set up fit matrix with linearly dependent columns
        A_tmp = self.A.copy()
        A_tmp[:, 0] = A_tmp[:, 1]

        # with check_condition
        fit(A_tmp, self.y, fit_method='least-squares', check_condition=True)

        # without check_condition
        fit(A_tmp, self.y, fit_method='least-squares', check_condition=False)

    def test_other_fit_methods(self):
        """Tests fit methods which are not run via available_fit_methods."""

        # lasso with alpha
        res = fit(self.A, self.y, fit_method='lasso', alpha=1e-5)
        self.assertIsNotNone(res['parameters'])

        # elasticnet with alpha
        res = fit(self.A, self.y, fit_method='elasticnet', alpha=1e-5)
        self.assertIsNotNone(res['parameters'])

        # rfe with n_features
        kwargs = dict(n_features=int(0.5*self.n_cols),
                      step=0.12, estimator='split-bregman',
                      final_estimator='rfe')
        res = fit(self.A, self.y, fit_method='rfe', **kwargs)
        self.assertIsNotNone(res['parameters'])
        self.assertEqual(len(res['parameters']), self.n_cols)
        self.assertEqual(sum(res['features']), kwargs['n_features'])

    def test_fit_with_invalid_fit_method(self):
        """Tests correct raise with unavailable fit_method."""
        bad_fit_methods = ['asd', '123', 42, ['lasso']]
        for fit_method in bad_fit_methods:
            with self.assertRaises(ValueError):
                fit(self.A, self.y, fit_method=fit_method)

    def test_ardr_line_scan(self):
        """Tests all available fit_methods."""

        res = fit(self.A, self.y, fit_method='ardr', line_scan=True)
        self.assertLess(np.linalg.norm(self.x - res['parameters']), 0.2)

        res = fit(self.A, self.y, fit_method='ardr', line_scan=True, threshold_lambdas=[1e4, 1e6])
        self.assertLess(np.linalg.norm(self.x - res['parameters']), 0.2)

        res = fit(self.A, self.y, fit_method='ardr', line_scan=True, cv_splits=5)
        self.assertLess(np.linalg.norm(self.x - res['parameters']), 0.2)

    def test_standardize(self):
        """ Tests that standardize kwarg works """
        res1 = fit(self.A, self.y, fit_method='least-squares', standardize=True)
        res2 = fit(self.A, self.y, fit_method='least-squares', standardize=False)
        np.testing.assert_almost_equal(res1['parameters'], res2['parameters'])


if __name__ == '__main__':
    unittest.main()
