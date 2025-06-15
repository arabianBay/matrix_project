import numpy as np
import time

N = 1024  # Matrix size

# Generate random matrices
A = np.random.randint(0, 10, (N, N))
B = np.random.randint(0, 10, (N, N))

# Measure execution time for the serial approach
start_time = time.time()
C_serial = np.dot(A, B)
serial_time = time.time() - start_time

print(f"Serial Execution Time: {serial_time:.6f} seconds")