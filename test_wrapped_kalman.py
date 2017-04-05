import matplotlib.pyplot as plt
from wrappedkalman import WrappedKalman
from numpy import matrix, array, pi, cos, arange, where, abs, diff, nan
from numpy.random import normal


class AzimuthSim:
    def __init__(self):
        # Simulates a noisy moving bearing measurement moving around the unit circle
        self.theta = 0
        self.actual = []
        self.measured = []
        self.t = 0

    def wrap(self, x):
        return ((x + pi) % (2 * pi)) - pi

    def step(self):
        self.theta = 2*pi * cos(1.2 * pi*self.t/200.0)
        self.actual.append(self.wrap(self.theta))
        self.measured.append(self.wrap(self.theta+normal(0, .5)))
        self.t += 1
        return self.measured[-1]


def main():
    az = AzimuthSim()

    x0 = 0.0
    var = 0.5
    cov = matrix([[.01, 0], [0, .001]])
    dt = 1.0

    km = WrappedKalman(x0, dt, cov, var)
    t = 200

    for _ in range(t):
        km.step(az.step())

    ax = plt.subplot(111)  #  projection='polar'
    actual = array(az.actual)
    actual[where(abs(diff(actual)) >= pi)[0]]=nan
    ax.plot(arange(t), actual,  label='true')
    ax.plot(arange(t), array(az.measured), 'x',  label='measure')
    theta = array([i[0, 0] for i in km.predictions])
    theta[where(abs(diff(theta)) >= pi)[0]] = nan
    ax.plot(arange(t), theta[1:], label='kalman')
    plt.ylim(- pi,  pi)
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
