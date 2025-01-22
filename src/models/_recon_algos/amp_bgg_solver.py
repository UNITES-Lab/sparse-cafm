"""
Module to perform the computational Approximate Message Passing
(AMP) algorithm. The implemented function 'camp' is the classical
approach with a Bernoulli-Gaussian input channel prior and a
Gaussian output channel prior.

The implementation serves as a reference implementation with close
link to the algorithm, which costs in computation time.

(C) Torben Larsen, Christian Schou Oxvig, Thomas Arildsen
    Aalborg University, Denmark
    {tl, cso, tha}@es.aau.dk

Python >2.6 and >3.3 compliant.

"""


import numpy as np


def _iBGoG(y_bar, y_hat, ipars):
    """Prior estimator - Bernoulli-Gaussian input and Gaussian output.
    
    """
    # Input channel parameters
    theta_bar = ipars['theta_bar']
    theta_hat = ipars['theta_hat']
    rho = ipars['rho']

    # Common parameters
    common_denominator = y_hat + theta_hat
    M_bar = (y_hat * theta_bar + y_bar * theta_hat) / common_denominator
    V_hat = y_hat * theta_hat / common_denominator

    # Common part
    z = (1. - rho) / rho * np.sqrt(theta_hat / V_hat)
    z *= np.exp(-.5 * ( y_bar ** 2 / y_hat
         - (theta_bar - y_bar) ** 2 / (y_hat + theta_hat) ))

    # Estimated mean and variance
    a = M_bar / (z + 1)
    v = z * a ** 2 + V_hat / (z + 1)

    return a, v




def camp(y, F, d_s, ipars, T=300, sol_tol=1.0E-2):
    """Computational Approximate Message Passing (CAMP).
    Aims to solve for minimum 2-norm of the residual error,
    r = y - np.dot(F, a_ast). The iterative AMP algorithm is applied where
    the stopping criteria is one of:
      1) Number of iterations t == T.
      2) ||r||_2 < sol_tol * ||y||_2

    Arguments:
      y : numpy.ndarray, shape=(N), dtype=float
        Measurement array.
      F : numpy-ndarray, shape=(M, N), dtype=float
        System matrix.
      d_s : float
        Initial stepsize.
      ipars : dict
        Key/value pairs include: ipars['theta_bar'] = theta_bar,
        ipars['theta_hat'] = theta_hat and ipars['rho'] = rho.
      T : int
        Maximum number of iterations.
      sol_tol : float
        Tolerance on the residual error.

      a_est_col : numpy.ndarray, shape=(N, 1), dtype=float
        Column vector of estimated means - should be close to x.
      v_est_col : numpy.ndarray, shape=(N, 1), dtype=float
        Column vector of estimated variances.

    """
    # Initialize arrays
    M, N = F.shape
    a_est, v_est = np.zeros(N), np.ones(N)
    w, v = y.copy(), np.ones(M)
    F_2 = F * F

    # Iterate for solution
    for t in range(T):
        # Prepare the message passing elements of the algorithm
        w = np.dot(F, a_est) - np.dot(F_2, v_est) * (y - w) / (d_s + v)
        v = np.dot(F_2, v_est)
        y_hat = 1.0 / np.dot(F_2.T, 1.0/(d_s + v))
        y_bar = a_est + y_hat * np.dot(F.T, (y - w)/(d_s + v))

        # Update prior estimate
        a_est, v_est = _iBGoG(y_bar, y_hat, ipars)

        # Uncomment this line to learn \Delta iteratively.
        d_s = ( d_s * np.sum(((y - w) / (d_s + v)) ** 2)
                / np.sum(1. / (d_s + v)) )

        # Residual; break loop if residual is smaller than requested
        r = y - np.dot(F, a_est)
        if np.linalg.norm(r, ord=2) < sol_tol * np.linalg.norm(y, ord=2):
            break

    # Reshape estimates to column vectors
    a_est_col = a_est.reshape((len(a_est), 1))
    v_est_col = v_est.reshape((len(a_est), 1))

    return a_est_col, v_est_col
