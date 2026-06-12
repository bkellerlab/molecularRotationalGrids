import numpy as np
from scipy.sparse.linalg import eigs
from scipy import sparse
import time
import csv
from datetime import timedelta

# loading
my_matrix = sparse.load_npz("rate_matrix.npz")

start_time = time.time()
eigenval, eigenvec = eigs(my_matrix.T, k=6, tol=0, maxiter=100000, which="SR",sigma=0)
end_time = time.time()

# save the time to file
elapsed_seconds = end_time - start_time
elapsed_hms = str(timedelta(seconds=int(elapsed_seconds)))  # Convert to h:m:s

csv_file = "timing_results.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time (h:m:s)", "Time (seconds)"])
    writer.writerow([elapsed_hms, elapsed_seconds])

# if imaginary eigenvectors or eigenvalues, raise error
if not np.allclose(eigenvec.imag.max(),0,rtol=1e-3,atol=1e-5) or not np.allclose(eigenval.imag.max(),0,
        rtol=1e-3,atol=1e-5):
    print(f"Complex values for eigenvectors and/or eigenvalues: {eigenvec}, {eigenval}")
eigenvec = eigenvec.real
eigenval = eigenval.real
# sort eigenvectors according to their eigenvalues
idx = eigenval.argsort()[::-1]
eigenval = eigenval[idx]
eigenvec = eigenvec[:, idx]

# saving to file
np.save("eigenvalues.npy",np.array(eigenval))
np.save("eigenvectors.npy",np.array(eigenvec))
