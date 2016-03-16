from __future__ import absolute_import, division, print_function

from functools import partial

import numpy as np
from scipy.spatial import KDTree
from scipy.special import psi
from joblib import Parallel, delayed


def estimate(X, Y, k=None, n_jobs=1):
    """ Estimate univesal k-NN divergence.

    Parameters
    ----------
        X, Y:
            2-dimensional array where each row is a sample.

        k:
            k-NN to be used. None for adaptive choice.

        n_jobs:
            number of jobs to run in parallel. Python 2 may only work with
            ``n_jobs=1``.
    """

    if not (isinstance(k, int) or k is None):
        raise ValueError('k has incorrect type.')
    if k is not None and k <= 0:
        raise ValueError('k cannot be <= 0')
    X = np.array(X)
    Y = np.array(Y)
    if len(X.shape) != 2 or len(Y.shape) != 2:
        raise ValueError('X or Y has incorrect dimension.')
    if X.shape[0] <= 1 or Y.shape[0] <= 1:
        raise ValueError('number of samples is not sufficient.')
    if X.shape[1] != Y.shape[1]:
        raise ValueError('numbers of columns of X and Y are different.')
    d = X.shape[1]
    n = X.shape[0]
    m = Y.shape[0]

    X_tree = KDTree(X)
    Y_tree = KDTree(Y)

    P = Parallel(n_jobs)
    nhu_ro = P(delayed(__calc_nu_rho)(x, X_tree, Y_tree, k) for x in X)
    r = (d / n) * sum(nhu_ro) + np.log(m / (n - 1))
    if k is None:
        l_k = P(delayed(__calc_l_k)(x, X_tree, Y_tree) for x in X)
        r += (1 / n) * sum(l_k)
    return r


def __get_epsilon(a, X_tree, Y_tree):
    offset_X = len([None for x in X_tree.data if (x == np.array(a)).all()])
    offset_Y = len([None for y in Y_tree.data if (y == np.array(a)).all()])
    rho_d, _ = X_tree.query([a], offset_X+1)
    nu_d, _ = Y_tree.query([a], offset_Y+1)
    rho_d = rho_d[0] if offset_X == 0 else rho_d[0][-1]
    nu_d = nu_d[0] if offset_Y == 0 else nu_d[0][-1]
    return max(rho_d, nu_d) + 0.5 ** 40


def __get_epsilon_sample_num(a, tree, X_tree, Y_tree, default_offset=0):
    e = __get_epsilon(a, X_tree, Y_tree)
    return len(tree.query_ball_point(a, e)) - default_offset


def __get_distance(a, tree, X_tree, Y_tree, k, default_offset):
    if k is None:
        k_ = __get_epsilon_sample_num(a, tree, X_tree, Y_tree)
    else:
        k_ = k + default_offset
    d, _ = tree.query([a], k_)
    return d[0] if k_ == 1 else d[0][-1]


def __calc_nu_rho(x, X_tree, Y_tree, k):
    rho = partial(__get_distance,
                  tree=X_tree, X_tree=X_tree, Y_tree=Y_tree,
                  k=k, default_offset=1)
    nu = partial(__get_distance,
                 tree=Y_tree, X_tree=X_tree, Y_tree=Y_tree,
                 k=k, default_offset=0)
    return np.log(nu(x) / rho(x))


def __calc_l_k(x, X_tree, Y_tree):
    _l = partial(__get_epsilon_sample_num,
                 tree=X_tree, X_tree=X_tree, Y_tree=Y_tree, default_offset=1)
    _k = partial(__get_epsilon_sample_num,
                 tree=Y_tree, X_tree=X_tree, Y_tree=Y_tree, default_offset=0)
    return psi(_l(x)) - psi(_k(x))
