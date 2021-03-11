import unittest

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing._data import _handle_zeros_in_scale


class TestNormalization(unittest.TestCase):
    def setUp(self):
        diabetes = load_diabetes()
        self.X = diabetes.data
        self.y = diabetes.target
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        self.mean_global = scaler.mean_
        self.var_global = scaler.var_
        self.scale_global = scaler.scale_
        self.X1 = self.X[:50]
        self.X2 = self.X[50:]
        m1 = np.mean(self.X1)
        var1 = np.var(self.X1)
        m2 = np.mean(self.X2)
        var2 = np.var(self.X2)
        self.mean_fed = (m1*len(self.X1) + m2*len(self.X2)) / (len(self.X1) + len(self.X2))
        self.var_fed = ((var1*len(self.X1) + var2*len(self.X2)) / (len(self.X1) + len(self.X2)))
        self.scale_fed = _handle_zeros_in_scale(np.sqrt(self.var_fed))

    def test_normalization(self):
        np.testing.assert_array_almost_equal(self.mean_global, self.mean_fed, decimal=12)
        np.testing.assert_array_almost_equal(self.var_global, self.var_fed, decimal=5)
        np.testing.assert_array_almost_equal(self.scale_global, self.scale_fed, decimal=4)


if __name__ == "__main__":
    unittest.main()
