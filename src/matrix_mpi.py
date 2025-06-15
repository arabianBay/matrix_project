import sys
import os
import time


# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.matrix_utils import generate_matrix
from mpi4py import MPI
import numpy as np

N = 4  # Matrix size

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Root initializes matrices
if rank == 0:
    A = generate_matrix(N)
    B = generate_matrix(N)
    
    # Serial execution benchmark
    start_serial = time.time()
    C_serial = np.dot(A, B)
    serial_time = time.time() - start_serial

else:
    A = np.empty((N, N), dtype=int)
    B = np.empty((N, N), dtype=int)
    serial_time = None

# Broadcast serial_time to all processes
serial_time = comm.bcast(serial_time, root=0)

# Broadcast matrices
comm.Bcast(A, root=0)
comm.Bcast(B, root=0)

# Divide workload
rows_per_process = N // size
start_row = rank * rows_per_process
end_row = (rank + 1) * rows_per_process

# Start timing
comm.Barrier()  # Sync all processes before timing
start_time = MPI.Wtime()


# Compute local multiplication
C_local = np.zeros((rows_per_process, N), dtype=int)
for i in range(start_row, end_row):
    for j in range(N):
        C_local[i - start_row, j] = np.dot(A[i, :], B[:, j])

# Gather results
C = None
if rank == 0:
    C = np.zeros((N, N), dtype=int)

comm.Gather(C_local, C, root=0)

# End timing
comm.Barrier()
end_time = MPI.Wtime()

# Print results
if rank == 0:
    print("Matrix A:\n", A)
    print("\nMatrix B:\n", B)
    print("\nResultant Matrix C:\n", C)
    mpi_time = end_time - start_time
    print(f"MPI Execution Time: {mpi_time:.6f} seconds")
    print(f"Speedup: {serial_time / mpi_time:.2f}x (vs Serial)")

MPI.Finalize()
