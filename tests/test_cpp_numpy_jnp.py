# The Python/Numpy implementation
from py_pathnorm_proxmap import np_pathnorm_proxmap as np_prox
# The Python/JAX implementation
from py_pathnorm_proxmap import jax_pathnorm_proxmap as jnp_prox
# The C++/Eigen implementation
import pathnorm_proxmap as cpp_prox

def test_main():
    assert m.__version__ == "0.0.1"
    assert m.add(1, 2) == 3
    assert m.subtract(1, 2) == -1
