import numpy as np
from tqdm import trange

EPS = 2.0**-52


def NMF(V,
        W,
        H,
        B=10,
        num_iter=10,
        params=None,
        cost_func='Music',
        fixW=False):
    """Given a non-negative matrix V, find non-negative templates W and
    activations H that approximate V.

    References
    ----------
    [1] Lee, DD & Seung, HS. "Algorithms for Non-negative Matrix Factorization"

    [2] Andrzej Cichocki, Rafal Zdunek, Anh Huy Phan, and Shun-ichi Amari
    "Nonnegative Matrix and Tensor Factorizations: Applications to
    Exploratory Multi-Way Data Analysis and Blind Source Separation"
    John Wiley and Sons, 2009.

    [3] D. Jeong, T. Kwon, and J. Nam, “Note-Intensity Estimation of Piano
    Recordings Using Coarsely Aligned MIDI Score,” Journal of the Audio
    Engineering Society, vol. 68, no. 1/2, pp. 34--47, 2020.

    Parameters
    ----------
    V: array-like
        K x M non-negative matrix to be factorized

    cost_func : str
        Cost function used for the optimization, currently supported are:
          'EucDdist' for Euclidean Distance
          'KLDiv' for Kullback Leibler Divergence
          'ISDiv' for Itakura Saito Divergence
          'Music' for score-informed music applications [3]

    num_iter : int
        Number of iterations the algorithm will run.

    W  : np.ndarray
        The initial W modified in place

    H : np.ndarray
        The initial H modified in place

    fixW : bool
        If True, W is not updated

    params : dict
        parameters for `Music` updates with these names:
            a1, a2, a3, b1, b2, Mh, Mw

        `Mh` and `Mw` *MUST* be provided, the others can miss and in that case
        the following are used [3]:
            a1, a2, a3, b1, b2 = 30, 1, 100, 10, 1000

    B : int
        the number of basis for template
    """

    # get important params
    K, M = V.shape
    L = num_iter
    if cost_func == 'Music':
        # default ones
        a1, a2, a3, b1, b2, Mh, Mw = 30, 1, 100, 10, 1000, None, None
        if params:
            # load dictionary
            locals().update(params)
            if 'Mh' not in params or 'Mw' not in params:
                raise RuntimeError("Mh and Mw *MUST* be provided")
        else:
            raise RuntimeError("Please, provide params for `Music` cost function")

    # create helper matrix of all ones
    onesMatrix = np.ones((K, M))

    # normalize to unit sum
    V /= (EPS + V.sum())

    # main iterations
    for iter in trange(L, desc='NMF:'):

        # compute approximation
        Lambda = EPS + W @ H

        # switch between pre-defined update rules
        if cost_func == 'EucDist':
            # euclidean update rules
            if not fixW:
                W *= (V @ H.T / (Lambda @ H.T + EPS))

            H *= (W.T @ V / (W.T @ Lambda + EPS))

        elif cost_func == 'KLDiv':
            # Kullback Leibler divergence update rules
            if not fixW:
                W *= ((V / Lambda) @ H.T) / (onesMatrix @ H.T + EPS)

            H *= (W.T @ (V / Lambda)) / (W.T @ onesMatrix + EPS)

        elif cost_func == 'ISDiv':
            # Itakura Saito divergence update rules
            if not fixW:
                W *= ((Lambda**-2 * V) @ H.T) / ((Lambda**-1) @ H.T + EPS)

            H *= (W.T @ (Lambda**-2 * V)) / (W.T @ (Lambda**-1) + EPS)

        elif cost_func == 'Music':
            # update rules for music score-informed applications

            if not fixW:
                W_indicator = np.zeros_like(W)
                W_indicator[:, ::B] += W
                numW = (V / Lambda) @ H.T
                numW[1:] += 2 * b1 * W_indicator[1:]
                numW[:-1] += 2 * b1 * W_indicator[:-1] + b2 * Mw

                W *= numW / (onesMatrix @ H.T + EPS + b2 +
                             4 * b1 * W_indicator)

            numH = W.T @ (V / Lambda) + a1 * Mh
            numH[:, B:] += a2 * H[:, B:]
            numH[:, :-B] += H[:, :-B]
            H *= numH / (W.T @ onesMatrix + a1 + a3 + 4 * a2 * H)

        else:
            raise ValueError('Unknown cost function')

        # normalize templates to unit sum
        if not fixW:
            normVec = W.sum(axis=0)
            W *= 1.0 / (EPS + normVec)
