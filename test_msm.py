import time
import numpy as np
from ecpy.curves import Curve, Point
from utils import generate_points, manual_msm
from pippenger import pip_msm

def test_msm(n=10000):
    st = time.time()
    es, gs = generate_points(n)
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

if __name__ == '__main__':
    test_msm(1000)
