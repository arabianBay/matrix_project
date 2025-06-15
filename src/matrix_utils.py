import numpy as np

def generate_matrix(size, seed=42):
    """Generate a random matrix of given size."""
    np.random.seed(seed)
    return np.random.randint(0, 10, (size, size))

def serial_matrix_multiplication(A, B):
    """Standard matrix multiplication without MPI."""
    return np.dot(A, B)