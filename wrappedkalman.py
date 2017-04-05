"""
Python implementation of the wrapped kalman filter for bearing tracking 

The wrapped Kalman filter infers the hidden state of a linear 
determinate system where the state lies on the unit circle. 


Algorithm is described in:
    "A Wrapped Kalman Filter for Azimuthal Speaker Tracking"
    IEEE Signal Processing Letters, December 2013, Volume: 20 Issue: 12
    Pages 1257-1260 (Johannes Traa and Paris Smaragdis)
"""

from numpy import matrix, square, exp, sqrt, pi, array, eye
from numpy.linalg import inv


class WrappedKalman:
    def __init__(self, x0, dt, cov, var, l=1, v0=0.0):
        """
        :param x0: float
        :param dt: float
        :param cov: numpy.matrixlib.defmatrix.matrix
        :param var: float
        :param l: int
        :param v0: float
        """
        self.A = matrix([[1, dt], [0, 1]])  # state transition matrix
        self.C = cov    # state covariance matrix
        self.cov = cov  # process noise
        self.B = matrix([1, 0])     # measurement matrix
        self.var = var      # measurement noise
        self.L = range(-l, l + 1)   # replicates
        self.state = matrix([[x0], [v0]])    # current state
        self.predictions = [self.state]     # historical states

    def wrap(self, x):
        """
        :param x: float
        :return: float
        """
        # returns the value of an angle wrapped around the unit circle
        return ((x + pi) % (2 * pi)) - pi

    def step(self, y):
        """
        :param y: float
        :return: int
        """
        # prediction step
        # state prediction
        state_est = self.A * self.state
        # wrap predicted state
        state_est[0, 0] = self.wrap(state_est[0, 0])
        # estimate covariance
        c_est = self.A * self.C * self.A.T + self.cov

        # innovation step
        # formulate normal pdf
        N = lambda x: (1 / sqrt(2 * pi * self.var)) * exp(-square(x - state_est[0, 0]) / (2 * self.var))
        # calculate partial probabilities
        p_yl = array([N(y + 2 * pi * l) for l in self.L])
        # calculate conditional probabilities
        p_y = p_yl / sum(p_yl)
        # calculate innovation conditional on l
        g_l = array([((y + 2 * pi * l)-state_est[0, 0]) for l in self.L])
        # calculate weighted innovation
        g = sum(g_l * p_y)

        # correction step
        # calculate Kalman gain
        K = (c_est * self.B.T) * inv(self.B * c_est * self.B.T + self.var)
        # update state
        self.state = state_est + K * g
        # wrap predicted state
        self.state[0, 0] = self.wrap(self.state[0, 0])
        # update covariance
        self.cov = (eye(self.cov.shape[0]) - K * self.B) * c_est

        # add current state to predictions
        self.predictions.append(self.state)
