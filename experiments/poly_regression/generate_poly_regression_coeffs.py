import numpy as np

def relu(x):
    return np.maximum(0,x)

def silu(x):
    return x / (1 + np.exp(-x))

def generate_coeffs(func, R, granularity, order):
    steps = int(2*R/granularity)
    xs = np.linspace(-R,R,steps)
    ys = func(xs)
    coeffs = np.polyfit(xs, ys, deg=order)[::-1]
    return coeffs
