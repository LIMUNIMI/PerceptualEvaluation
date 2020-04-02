"""
Fast computation of distances with float32 and cython. Distancs matrix computed
with thread parallelization!
"""
import numpy as np
cimport cython
from cython.parallel cimport prange
from libc.math cimport sqrt
from libc.math cimport fabs


cdef float eps = 1e-15

cdef float[:] sub(float[:] A, float c) nogil:
    for i in range(len(A)):
        A[i] -= c

    return A

cdef float norm(float[:] A) nogil:
    cdef float out = 0
    for i in range(len(A)):
        out += A[i]**2

    return sqrt(out) + eps

cdef float dot(float[:] A, float[:] B) nogil:
    cdef float out = 0
    for i in range(len(A)):
        out += A[i] * B[i]

    return out

cdef float mean(float[:] A) nogil:
    cdef float out = 0
    for i in range(len(A)):
        out += A[i]

    return out / len(A)

cdef float cosine_distance(float[:] A, float[:] B) nogil:
    return 1 - dot(A, B) / (norm(A) * norm(B))

cdef float hamming(float[:] A, float[:] B) nogil:
    cdef float out = 0
    for i in range(len(A)):
        if A[i] != B[i]:
            out += 1
    return out

cdef float normalized_minkowski(float[:] A, float[:] B, float p) nogil:
    cdef float out = 0
    cdef float normA = 0
    cdef float normB = 0
    for i in range(len(A)):
        out += fabs(A[i] - B[i])**p
        normA += fabs(A[i])**p
        normB += fabs(B[i])**p
    normA = normA ** (1.0/p) + eps
    normB = normB ** (1.0/p) + eps
    return out**(1.0/p) / max(normA, normB)
    # return out**(1.0/p)

cdef float braycurtis(float[:] A, float[:] B) nogil:
    cdef float num = 0
    cdef float den = 0

    for i in range(len(A)):
        num += fabs(A[i] - B[i])
        den += fabs(A[i] + B[i])

    return num/(den + eps)

cdef float canberra(float[:] A, float[:] B) nogil:
    cdef float out = 0

    for i in range(len(A)):
        out += fabs(A[i] - B[i]) / (fabs(A[i]) + fabs(B[i]) + eps)

    return out

cdef float chebyshev(float[:] A, float[:] B) nogil:
    cdef float out = -1

    for i in range(len(A)):
        out = max(out, fabs(A[i] - B[i]))

    return out

cdef float correlation(float[:] A, float[:] B) nogil:
    A = sub(A, mean(A))
    B = sub(B, mean(B))
    return cosine_distance(A, B)


def cdist(float[:, :] A, float[:, :] B, str metric='cosine', float p=1):
    """
    Compute distance matrix with parallel threading (without GIL).

    Arguments
    ---
    * `A` : array of array -> float32 (np.array)
        First collection of samples. Features are the last dimension.
    * `B` : array of array -> float32 (np.array)
        Second collection of samples. Features are the last dimension.
    * `metric` : str
        A string indicating the metric that should be used.
        Available metrics:
        - 'cosine' : 1 - cosine similarity
        - 'minkowski' : minkowski normalized to 1
        - 'hamming' : number of different entries
        - 'braycurtis' : bray-curtis distance
        - 'canberra' : canberra distance
        - 'chebyshev' : chebyshev distance
        - 'correlation' : correlation distance
    * `p` : float32
        A value representing the `p` used for minkowski

    Returns
    ---
    * array of array :
        shape: (A.shape[0], B.shape[0])
    """
    cdef int a_cols = A.shape[1]
    cdef int b_cols = B.shape[1]
    cdef float[:, :] out = np.empty((a_cols, b_cols), dtype=np.float32)
    cdef int i = 0
    cdef int j = 0

    for i in prange(a_cols, nogil=True):
        for j in range(b_cols):
            if metric == 'cosine':
                out[i, j] = cosine_distance(A[:, i], B[:, j])
            elif metric == 'hamming':
                out[i, j] = hamming(A[:, i], B[:, j])
            elif metric == 'minkowski':
                out[i, j] = normalized_minkowski(A[:, i], B[:, j], p)
            elif metric == 'braycurtis':
                out[i, j] = braycurtis(A[:, i], B[:, j])
            elif metric == 'canberra':
                out[i, j] = canberra(A[:, i], B[:, j])
            elif metric == 'chebyshev':
                out[i, j] = chebyshev(A[:, i], B[:, j])
            elif metric == 'correlation':
                out[i, j] = correlation(A[:, i], B[:, j])

    return out
