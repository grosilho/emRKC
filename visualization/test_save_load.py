import numpy as np

file = "file.npy"
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8, 9], [10, 11, 12]])

with open(file, "wb") as f:
    np.save(f, A)
    np.save(f, B)

with open(file, "rb") as f:
    A = np.load(f)
    B = np.load(f)
