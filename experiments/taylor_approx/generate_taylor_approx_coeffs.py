import numpy as np
import jax.numpy as jnp
from jax import grad, vmap


def relu(x):
    return jnp.maximum(0,x)

def silu(x):
    return x / (1 + jnp.exp(-x))

def generate_coeffs(func, a, order):
    a = float(a)
    coeffs = []
    for i in range(order+1):
        curr_coeff = func(a) / float(np.math.factorial(i))
        coeffs.append(curr_coeff)
        func = grad(func)
    return coeffs

#print(generate_coeffs(silu, 0, 2))
