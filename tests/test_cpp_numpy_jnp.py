import time
from datetime import timedelta

# The Python/Numpy implementation
from py_pathnorm_proxmap.np_pathnorm_proxmap import PathNormProximalMapping as ProxNP
# The Python/JAX implementation
from py_pathnorm_proxmap.jax_pathnorm_proxmap import PathNormProximalMapping as ProxJAX
# The C++/Eigen implementation
from pathnorm_proxmap import PathNormProximalMap as ProxCPP

import numpy as np
import jax.numpy as jnp


def test_np():
    np.random.seed(53)


X = np.random.randn(150, 400)
Y = np.random.randn(150, 200)

# def test_main():
if __name__ == '__main__':
    np.random.seed(53)
    X = np.random.randn(100, 120)
    Y = np.random.randn(100, 50)
    jnp_XY = (jnp.array(X), jnp.array(Y))
    la = 1e-2

    print_stats = lambda times: f"mean: {sum(times) / len(times)}, min: {min(times)}, max: {max(times)}"

    print('Running NumPy benchmark')
    # Warming
    for _ in range(10):
        np_VW = ProxNP.prox_map((X, Y), la)

    np_time = []
    # Go
    for _ in range(100):
        start_time = time.monotonic()
        np_VW = ProxNP.prox_map((X, Y), la)
        np_time.append(time.monotonic() - start_time)
    print(print_stats(np_time))

    print('Running JAX benchmark')
    # Warming
    for _ in range(10):
        jnp_VW = ProxJAX.prox_map(jnp_XY, la)

    jx_time = []
    # Go
    for _ in range(100):
        start_time = time.monotonic()
        jnpV, jnpW = ProxJAX.prox_map(jnp_XY, la)
        jnpV.block_until_ready()
        jnpW.block_until_ready()
        jx_time.append(time.monotonic() - start_time)
    print(print_stats(jx_time))

    print('Running C++ benchmark')
    # Warming
    for _ in range(10):
        cpp_prox = ProxCPP(X, Y, la)
        cpp_prox.run()
        cpp_VW = (cpp_prox.V, cpp_prox.W)

    cpp_time = []
    # Go
    for _ in range(100):
        start_time = time.monotonic()
        cpp_prox = ProxCPP(X, Y, la)
        cpp_prox.run()
        cpp_VW = (cpp_prox.V, cpp_prox.W)
        cpp_time.append(time.monotonic() - start_time)
    print(print_stats(cpp_time))

    assert np_VW[0].shape == jnp_VW[0].shape and np_VW[1].shape == jnp_VW[1].shape
    assert np_VW[0].shape == cpp_VW[0].shape and np_VW[1].shape == cpp_VW[1].shape

    assert np.allclose(np_VW[0], jnp_VW[0]), f"np(V) != jnp(V), np(V)={np_VW[0]}, jnp(V)={jnp_VW[0]}"
    assert np.allclose(np_VW[1], jnp_VW[1]), f"np(W) != jnp(W), np(W)={np_VW[1]}, jnp(W)={jnp_VW[1]}"

    assert np.allclose(np_VW[0], cpp_VW[0]), f"np(V) != cpp(V), np(V)={np_VW[0]}, cpp(V)={cpp_VW[0]}"
    assert np.allclose(np_VW[1], cpp_VW[1]), f"np(W) != cpp(W), np(W)={np_VW[1]}, cpp(W)={cpp_VW[1]}"
