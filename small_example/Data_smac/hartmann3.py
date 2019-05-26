import numpy as np

def hartmann3(x):
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = np.array([[3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0],
                  [3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0]])
    P = 0.0001 * np.array([[3689, 1170, 2673],
                           [4699, 4387, 7470],
                           [1090, 8732, 5547],
                           [381, 5743, 8828]])
    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(3):
            internal_sum += A[i, j] * (x[j] - P[i, j]) ** 2
        external_sum += alpha[i] * np.exp(-internal_sum)
    
    return external_sum
