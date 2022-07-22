import time
import numpy as np
from utils import generate_points, manual_msm
from pippenger import pip_msm
from cuda_pippenger import cuda_pip_msm
import ec.ecp_functional as ecpf

def test_cuda_msm(n=10000):
    st = time.time()
    es, gs = generate_points('32bit', n)
    print('Time to gen points: {:.2f}s'.format(time.time() - st))
    es = np.abs(es) # TODO: handle negative case

    st = time.time()
    true_sum = pip_msm(es, gs, num_bits=32, c=8)
    t1 = time.time() - st

    st = time.time()
    psum = cuda_pip_msm(es, gs, num_bits=32, c=8)
    t2 = time.time() - st
    print("pip:", true_sum.x, true_sum.y)
    print("cuda pip:", psum[0], psum[1])
    assert(true_sum.x == psum[0] and true_sum.y == psum[1])
    print('CPU pip: {:.2f}s | GPU pip: {:.2f}s | Len: {}'.format(t1, t2, len(gs)))

if __name__ == '__main__':
    test_cuda_msm(1000)
