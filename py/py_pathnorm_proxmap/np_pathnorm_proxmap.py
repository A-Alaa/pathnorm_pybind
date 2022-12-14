"""."""

from functools import partial

import numpy as np


class PathNormProximalMapping:
    @staticmethod
    def zeros_like(*args, **kwargs):
        return np.zeros_like(*args, **kwargs)

    @staticmethod
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

        v_t = 0.5 * np.sum(np.square(vw[0] - a_xy[0]))
        w_t = 0.5 * np.sum(np.square(vw[1] - a_xy[1]))
        return v_t + w_t + lam * np.sum(vw[0]) * np.sum(vw[1])

    @staticmethod
    def preprocess_prox_params(XY):
        # Record the signs
        sign = np.sign(XY[0]), np.sign(XY[1])

        # The absolute
        a = np.abs(XY[0]), np.abs(XY[1])

        # Descending-sorter
        sorter = np.argsort(-a[0], axis=1), np.argsort(-a[1], axis=1)

        # To undo-sorting
        recover0 = np.argsort(sorter[0], axis=1)
        recover1 = np.argsort(sorter[1], axis=1)
        recover = (recover0, recover1)

        # Sort
        a0 = np.take_along_axis(a[0], sorter[0], axis=1)
        a1 = np.take_along_axis(a[1], sorter[1], axis=1)
        a = (a0, a1)

        # Pre-calculations
        c_a = np.cumsum(a[0], axis=1), np.cumsum(a[1], axis=1)

        return a, c_a, sign, recover

    @staticmethod
    def postprocess_prox_params(VW, sign, recover):
        V, W = np.vstack(VW[0]), np.vstack(VW[1])
        V = np.take_along_axis(V, recover[0], axis=1)
        W = np.take_along_axis(W, recover[1], axis=1)
        return (V * sign[0], W * sign[1])

    @staticmethod
    def vw_stationary(a_xy, ca_xy, s_vw, lambd):
        """
        Compute the stationary points v^(sv, sw), w^(sv, sw) using Eq(26)
        (Latorre et al.; 2020) for the pair (sv, sw) passed as tuple 's_vw'.
        Args:
            a_xy: A tuple containing np.abs(x) and np.abs(y) vectors, sorted
                descendingly.
            ca_xy: A tuple containing two precomputed cumulative summation vectors.
                For example:
                    ca_xy = (np.cumsum(np.abs(x)), np.cumsum(np.abs(y))).
            s_vw: A tuple containing the (v, w) coordinates.
            lambd: Proximal operator hyperparameter.
        """

        ax, ay = a_xy
        sv, sw = s_vw
        la = lambd
        la2 = lambd ** 2
        sumx = ca_xy[0][sv - 1] if sv > 0 else 0
        sumy = ca_xy[1][sw - 1] if sw > 0 else 0
        u = 1. / (1 - sv * sw * la2)

        v = ax + u * (la2 * sw * sumx - la * sumy)
        w = ay + u * (la2 * sv * sumy - la * sumx)

        # v[k], w[k] = 0 for
        v[sv:] = 0
        w[sw:] = 0
        return v, w

    @classmethod
    def reject(cls, a_xy, ca_xy, s_vw, lambd):
        """
        Check where conditions 1 & 2 in Lemma 18 (Latorre et al.; 2020) are violated.
        """
        la = lambd
        la2 = lambd ** 2
        if s_vw[0] * s_vw[1] > 1. / la2:
            # Condition 1 is violated.
            return True
        ax, ay = a_xy
        sv, sw = s_vw

        sumx = ca_xy[0][sv - 1] if sv > 0 else 0
        sumy = ca_xy[1][sw - 1] if sw > 0 else 0
        u = 1. / (1 - sv * sw * la2)

        # Beware that -ve idx k in NumPy will return the kth element from
        # the end, what we want here is to return zero instead at -ve idx!
        sv, sw = s_vw

        v_sv = (ax[sv - 1] + u * (la2 * sw * sumx - la * sumy)) if sv > 0 else 0
        w_sw = (ay[sw - 1] + u * (la2 * sv * sumy - la * sumx)) if sw > 0 else 0

        # Condition 2 is violated.
        return v_sv < 0 or w_sw < 0

    @classmethod
    def sparsity_pairs_mfb(cls, a_xy, ca_xy, lambd):
        """
        Find sparsity pairs on the maximal feasibility boundary, Algorithm 5 (Latorre et al.; 2020).
        """
        m = len(a_xy[1])
        p = len(a_xy[0])
        sv = 0
        sw = m
        s = set()
        maximal = True
        while sv <= p and sw >= 0:
            if cls.reject(a_xy, ca_xy, (sv, sw), lambd):
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
        hopt = np.inf

        for s_vw in cls.sparsity_pairs_mfb(a_xy, ca_xy, lam):
            vw = cls.vw_stationary(a_xy, ca_xy, s_vw, lam)
            h = cls.prox_obj(a_xy, vw, lam)
            if h < hopt:
                hopt = h
                vw_sol = vw

        probe_v = (cls.zeros_like(a_xy[0]), a_xy[1])
        if cls.prox_obj(a_xy, probe_v, lam) < hopt:
            vw_sol = probe_v

        probe_w = (a_xy[0], cls.zeros_like(a_xy[1]))
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
