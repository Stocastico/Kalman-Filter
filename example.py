import numpy as np
import matplotlib.pyplot as plt
from KalmanFilter import KalmanFilter


def example1():
    """
    Test Kalman Filter implementation
    :return:
    """
    dt = 1.0/60
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)

    x = np.linspace(-10, 10, 100)
    measurements = - (x**2 + 2*x - 2) + np.random.normal(0, 2, 100)
    kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
    predictions = []

    for z in measurements:
        predictions.append(np.dot(H,  kf.predict())[0])
        kf.update(z)

    plt.plot(range(len(measurements)), measurements, label='Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label='Kalman Filter Prediction')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    example1()