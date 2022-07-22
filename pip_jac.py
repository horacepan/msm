from ec.ecp_functional import add, add_jac, jac2aff, aff2jac, dbl_jac
from ec.ecp_functional import jac_add
import ec.ecp

JAC_INF = (1, 1, 0)
'''
Given the affine coordinates of the elliptic curve points,
we keep track of running totals for Pippenger in Jacobian coordinates
The conversion for affine -> jacobian is just tacking on an additional 1
for the Z coordinate so we don't need to preprocess our input points.
'''
def pip_msm_jac(es, gs, num_bits, c, a, q):
    '''
    Expect input points to be in jacobian form
    Return output in jacobian form
    es: scalars
    gs: points
    num_bits: bit length of scalars
    c: bit window size for pippengers algo
    a: param for curve in Weirstrass form
    q: prime field of curve
    Returns: affine coordinates of the resulting elliptic curve point
    '''
    nbuckets = num_bits // c
    two_c = int(2**c)
    window_sums = [JAC_INF] * (nbuckets + 1)

    for widx in range(nbuckets + 1):
        # Point.inf + g = g + Point.inf = g for all g \in curve
        right_shift = c * widx
        bucket = [JAC_INF] * (two_c)
        for e, g in zip(es, gs):
            b = (e >> right_shift) % two_c # bit shift, grab last c bits
            if b == 0:
                continue
            else:
                bucket[b] = jac_add(*bucket[b], g.x, g.y, 1, a, q)

        acc = JAC_INF
        running_sum = JAC_INF
        for j in range(len(bucket) - 1, 0, -1):
            running_sum = jac_add(*running_sum, *bucket[j], a, q)
            acc = jac_add(*acc, *running_sum, a, q)

        window_sums[widx] = acc

    total = JAC_INF
    for window in window_sums[::-1]:
        for _ in range(c):
            total = dbl_jac(*total, q, a)
        total = jac_add(*total, *window, a, q)

    # convert from jacobian -> affine
    aff = jac2aff(*total, q)
    return aff
