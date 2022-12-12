
from functools import partial

import jax
import jax.numpy as jnp


class PathNormProximalMapping:

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
        u = 1. / (1 - sv * sw * lambd**2)

        v = ax + u * (lambd**2 * sw * ca_x - lambd * ca_y)
        w = ay + u * (lambd**2 * sv * ca_y - lambd * ca_x)

        # v[k], w[k] = 0 for
        v = v.at[sv:].set(0)
        w = w.at[sw:].set(0)
        return v, w

    @classmethod
    def postprocess_prox_params(cls, VW, sign, recover):
        V, W = jnp.vstack(VW[0]), jnp.vstack(VW[1])
        V = jnp.take_along_axis(V, recover[0], axis=1)
        W = jnp.take_along_axis(W, recover[1], axis=1)
        return (V * sign[0], W * sign[1])

    @classmethod
    def reject(cls, a_xy, ca_xy, s_vw, lambd, cache):
        """
        Check if condition 1 in Lemma 18 (Latorre et al.; 2020) is violated.
        """

        if s_vw[0] * s_vw[1] > lambd**-2:
            return True

        if s_vw not in cache:
            cache[s_vw] = cls.vw_stationary(a_xy, ca_xy, s_vw, lambd)
        v, w = cache[s_vw]
        # Beware that -ve idx k in NumPy will return the kth element from
        # the end, what we want here is to return zero instead at -ve idx!
        sv, sw = s_vw
        v_sv = v[sv - 1] if sv > 0 else 0
        w_sw = w[sw - 1] if sw > 0 else 0

        return v_sv < 0 or w_sw < 0

    @classmethod
    def sparsity_pairs_MFB(cls, a_xy, ca_xy, lambd, c):
        """
        Find sparsity pairs, Algorithm 5 (Latorre et al.; 2020).
        """
        m = len(a_xy[1])
        p = len(a_xy[0])
        sv, sw = 0, m
        s = set()
        maximal = True
        while sv <= p and sw >= 0:
            if cls.reject(a_xy, ca_xy, (sv, sw), lambd, c):
                if maximal:
                    s.add((sv - 1, sw))
                    maximal = False
                sw -= 1
            else:
                sv += 1
                maximal = True
        if sv == p + 1:
            s.add((sv - 1, sw))
        return s

    @classmethod
    def prox_map_vec(cls, lam, a_xy, ca_xy):
        """
        Proximal mapping applied on m-th column of V and m-th row of W.
        """
        # Sparsity
        hopt = jnp.inf
        # Caching v, w
        vw_c = dict()

        for s_vw in cls.sparsity_pairs_MFB(a_xy, ca_xy, lam, vw_c):
            if s_vw not in vw_c:
                vw = cls.vw_stationary(a_xy, ca_xy, s_vw, lam)
            else:
                vw = vw_c[s_vw]

            h = cls.prox_obj(a_xy, vw, lam)
            if h < hopt:
                hopt = h
                vw_sol = vw

        probe_v = (jnp.zeros_like(a_xy[0]), a_xy[1])
        if cls.prox_obj(a_xy, probe_v, lam) < hopt:
            vw_sol = probe_v

        probe_w = (a_xy[0], jnp.zeros_like(a_xy[1]))
        if cls.prox_obj(a_xy, probe_w, lam) < hopt:
            vw_sol = probe_w

        return vw_sol

    @classmethod
    def prox_map(cls, XY, lam, scaling=1.0):
        # Return sorted matrices, and pre-calculate absolute and cum.sum.
        a, ca, sign, recover = cls.preprocess_prox_params(XY)
        # TODO: jax.vmap
        vec_fun = partial(cls.prox_map_vec, lam * scaling)
        VW = zip(*list(map(vec_fun, zip(*a), zip(*ca))))
        return cls.postprocess_prox_params(tuple(VW), sign, recover)