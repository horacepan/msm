import time
import numpy as np
from utils import generate_points, manual_msm
from pippenger import pip_msm
from pip_jac import pip_msm_jac

def test_msm(n=10000):
    np.random.seed(0)
    st = time.time()
    es, gs = generate_points('secp256k1', n)
    print('Time to gen points: {:.2f}s'.format(time.time() - st))
    es = np.abs(es) # TODO: handle negative case

    st = time.time()
    true_sum = manual_msm(es, gs)
    t1 = time.time() - st

    st = time.time()
    psum = pip_msm(es, gs, num_bits=256, c=8)
    t2 = time.time() - st
    assert(true_sum == psum)
    print('Native: {:.2f}s | Pip: {:.2f}s | Len: {}'.format(t1, t2, len(gs)))

    st = time.time()
    jsum = pip_msm_jac(es, gs, num_bits=256, c=8, a=gs[0].a, q=gs[0].q)
    t3 = time.time() - st
    assert(psum.x == jsum[0] and psum.y == jsum[1])
    print('n = {}'.format(n))
    print('Native: {:.2f}s | Pip: {:.2f}s | jac: {:.2f}s'.format(t1, t2, t3))

if __name__ == '__main__':
    test_msm(1000)
