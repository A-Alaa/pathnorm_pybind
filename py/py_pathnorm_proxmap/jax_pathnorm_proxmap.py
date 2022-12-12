from functools import partial

import jax
import jax.numpy as jnp

from .np_pathnorm_proxmap import PathNormProximalMapping as NumPyBaseProximalMapping


class PathNormProximalMapping(NumPyBaseProximalMapping):
    @staticmethod
    def zeros_like(*args, **kwargs):
        return jnp.zeros_like(*args, **kwargs)

    @staticmethod
    @jax.jit
    def prox_obj(a_xy, vw, lam):
        """
        Evaluate the proximal objective:
            ||v - x||^2 + ||w - y||^2 + lambda * ||v||_1 ||w||_1
        Args:
            a_xy: A tuple containing x and y vectors, sorted
                descendingly by their absolute magnitudes.
            vw: A tuple containing the two input vectors v and w.
            lam: Proximal operator hyperparameter.
        """

        v_t = 0.5 * jnp.sum(jnp.square(vw[0] - a_xy[0]))
        w_t = 0.5 * jnp.sum(jnp.square(vw[1] - a_xy[1]))
        return v_t + w_t + lam * jnp.sum(vw[0]) * jnp.sum(vw[1])

    @staticmethod
    @jax.jit
    def preprocess_prox_params(XY):
        # Record the signs
        sign = jnp.sign(XY[0]), jnp.sign(XY[1])

        # The absolute
        a = jnp.abs(XY[0]), jnp.abs(XY[1])

        # Descending-sorter
        sorter = jnp.argsort(-a[0], axis=1), jnp.argsort(-a[1], axis=1)

        # To undo-sorting
        recover0 = jnp.argsort(sorter[0], axis=1)
        recover1 = jnp.argsort(sorter[1], axis=1)
        recover = (recover0, recover1)

        # Sort
        a0 = jnp.take_along_axis(a[0], sorter[0], axis=1)
        a1 = jnp.take_along_axis(a[1], sorter[1], axis=1)
        a = (a0, a1)

        # Pre-calculations
        c_a = jnp.cumsum(a[0], axis=1), jnp.cumsum(a[1], axis=1)

        return a, c_a, sign, recover

    @staticmethod
    def vw_stationary(a_xy, ca_xy, s_vw, lambd):
        """
        Compute the stationary points v^(sv, sw), w^(sv, sw) using Eq(26)
        (Latorre et al.; 2020) for the pair (sv, sw) passed as tuple 's_vw'.
        Args:
            a_xy: A tuple containing jnp.abs(x) and jnp.abs(y) vectors, sorted
                descendingly.
            ca_xy: A tuple containing two precomputed cumulative summation vectors.
                For example:
                    ca_xy = (jnp.cumsum(jnp.abs(x)), jnp.cumsum(jnp.abs(y))).
            s_vw: A tuple containing the (v, w) coordinates.
            lambd: Proximal operator hyperparameter.
        """

        ax, ay = a_xy
        sv, sw = s_vw

        ca_x = ca_xy[0][sv - 1] if sv > 0 else 0
        ca_y = ca_xy[1][sw - 1] if sw > 0 else 0
        u = 1. / (1 - sv * sw * lambd ** 2)

        v = ax + u * (lambd ** 2 * sw * ca_x - lambd * ca_y)
        w = ay + u * (lambd ** 2 * sv * ca_y - lambd * ca_x)

        # v[k], w[k] = 0 for
        v = v.at[sv:].set(0)
        w = w.at[sw:].set(0)
        return v, w

    @staticmethod
    @jax.jit
    def postprocess_prox_params(VW, sign, recover):
        V, W = jnp.vstack(VW[0]), jnp.vstack(VW[1])
        V = jnp.take_along_axis(V, recover[0], axis=1)
        W = jnp.take_along_axis(W, recover[1], axis=1)
        return (V * sign[0], W * sign[1])
