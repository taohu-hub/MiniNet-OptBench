import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pyDOE import lhs


# generate random points on sphere
def generate_rand_points_on_sphere(n, num_points=1):
    points = np.random.randn(num_points, n)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points /= norms
    return points


def generate_sample_points(dimension, num_points, radius=1):
    points = lhs(dimension, samples=num_points)
    points = points * 2 * radius - radius  # Scale to [-radius, radius]
    return points


def quadratic_regression(X, y):
    """
    input:
    X -- ndarray; sample points, every row is a point
    y -- 1D array; corresponding function value
    """
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    n_features = X.shape[1]
    quadratic_only = X_poly[:, n_features:]
    model = LinearRegression(fit_intercept=False).fit(quadratic_only, y)

    params = model.coef_
    hessian = np.zeros((n_features, n_features))
    idx = 0
    for i in range(n_features):
        for j in range(i, n_features):
            hessian[i, j] += params[idx]
            hessian[j, i] += params[idx]
            idx += 1
    return hessian


# a function that is needed in the truncated CG
def solve_for_tau(s, p, tr_radius):
    # Coefficients for the quadratic equation
    a_coef = np.dot(p.T, p)
    b_coef = 2 * np.dot(s.T, p)
    c_coef = np.dot(s.T, s) - tr_radius**2

    # Calculate the discriminant
    discriminant = b_coef**2 - 4 * a_coef * c_coef  # it is definitely >=0
    if math.isnan(discriminant):
        print("The discriminant is a nan due to some numerical reasons, set tau to 0")
        tau = 0
        return tau
    elif discriminant < 0:
        print("The discriminant is negative due to numerical reaons")
        discriminant = 0

    # Calculate the possible value of tau
    if a_coef > 0:
        tau = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef)
    else:
        tau = 0
    return tau
