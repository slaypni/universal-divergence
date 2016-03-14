universal-divergence
--------------------

universal-divergence is a Python module for estimating divergence of two sets of samples generated from the two underlying distributions.
The theory of the estimator is based on `a paper
<http://www.princeton.edu/~verdu/reprints/WanKulVer.May2009.pdf>`_ written by Q.Wang et al [1]_.

Install
-------

::

  pip install universal-divergence

Example
-------

::

  from __futere__ import print_function

  import numpy as np
  from universal_divergence import estimate

  mean = [0, 0]
  cov = [[1, 0], [0, 10]]
  x = np.random.multivariate_normal(mean, cov, 100)
  y = np.random.multivariate_normal(mean, cov, 100)
  print(estimate(x, y))  # will be close to 0.0

  mean2 = [10, 0]
  cov2 = [[5, 0], [0, 5]]
  z = np.random.multivariate_normal(mean2, cov2, 100)
  print(estimate(x, z))  # will be bigger than 0.0

References
----------

.. [1] Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdú. "Divergence estimation for multidimensional densities via k-nearest-neighbor distances." Information Theory, IEEE Transactions on 55.5 (2009): 2392-2405.
