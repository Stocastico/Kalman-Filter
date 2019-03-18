import numpy as np
from numpy.linalg import inv


class KalmanFilter:
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        """
        Initialize the Kalman Filter
        :param F: Prediction matrix
        :param B: Control matrix
        :param H: Readings matrix
        :param Q: Uncertainty from environment
        :param R: Uncertainty covariance
        :param P: Covariance matrix
        :param x0: Initial state

        """
        if F is None or H is None:
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.B = 0 if B is None else B
        self.H = H
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0  # set current state
        self.x_pred = None
        self.P_pred = None
        self. K = None  # kalman gain

    def predict(self, u=0):
        """
        Predicts the next state
        :param u: control vector
        :return: The predicted measurement
        """
        self.x_pred = self.F @ self.x + np.dot(self.B, u)
        self.P_pred = self.F @ self.P @ self.F.T + self.Q
        return self.x_pred

    def update(self, z):
        """
        Updates the filter parameters
        :param z: observation
        :return:
        """
        S = inv(self.H @ (self.P @ self.H.T) + self.R)
        self.K = (self.P_pred @ self.H.T) @ S
        self.x = self.x_pred + self.K @ (z - self.H @ self.x_pred)
        self.P = self.P_pred - (self.K @ self.H) @ self.P_pred
