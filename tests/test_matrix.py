import unittest
import numpy as np
from src.matrix_utils import generate_matrix, serial_matrix_multiplication

class TestMatrixOperations(unittest.TestCase):
    def setUp(self):
        self.N = 4
        self.A = generate_matrix(self.N)
        self.B = generate_matrix(self.N)

    def test_serial_multiplication(self):
        """Ensure matrix multiplication produces correct results."""
        expected = np.dot(self.A, self.B)
        result = serial_matrix_multiplication(self.A, self.B)
        np.testing.assert_array_equal(result, expected)

if __name__ == "__main__":
    unittest.main()