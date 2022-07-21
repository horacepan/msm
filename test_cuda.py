import time
import numpy as np
from utils import generate_points, manual_msm
from cuda_pippenger import cuda_pip_msm

def test_cuda_msm(n=10000):
    st = time.time()
    es, gs = generate_points('32bit', n)
    print('Time to gen points: {:.2f}s'.format(time.time() - st))
    es = np.abs(es) # TODO: handle negative case

    st = time.time()
    true_sum = manual_msm(es, gs)
    t1 = time.time() - st
    t1 = None

    st = time.time()
    psum = cuda_pip_msm(es, gs, num_bits=32, c=8)
    t2 = time.time() - st
    assert(true_sum == psum)
    print('Native: {:.2f}s | Pip: {:.2f}s | Len: {}'.format(t1, t2, len(gs)))

if __name__ == '__main__':
    test_cuda_msm(1000)
