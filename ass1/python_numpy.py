# Backup of completed TODOs in python_numpy.ipynb

import numpy as np

A = np.array([[0,1,2],[3,4,5]]) # shape (2,3)
B = np.array([[1,1,1]]) # shape (1,3)
C = np.array([[-1,-1,-1],[1,1,1]]) # shape (2,3)

# TODO
# Create matrix "D" as A - B using broadcasting
D = A - B

# Create matrix "E" with shape (3,2) by reshaping C
E = np.reshape(C, (3, 2))

# Create matrix "F" with shape (2,2) by matrix multiplying "D" by "E"
F = np.dot(D, E)

assert(np.all(D == [[-1,0,1],[2,3,4]]))
assert(np.all(E == [[-1,-1],[-1,1],[1,1]]))
assert(np.all(F == [[2,2],[-1,5]]))