import numba.cuda as cuda
import numba
import numpy as np
import math

def cuda_pip_msm(scalars, points, num_bits, c):
    nbuckets = math.ceil(num_bits/c)
    two_c = int(2**c)

    @cuda.jit
    def pippengers(scalars, points, result):
        # Allocate local memory for the window buckets and
        # for the window sums
        buckets = numba.cuda.local.array((two_c,), numba.u4)
        windowSums = numba.cuda.local.array((nbuckets,), numba.u4)

        for windowIdx in range(nbuckets):
            rightShift = two_c * windowIdx
            for s, p in zip(scalars, points):
                bucketIdx = (s >> rightShift) % two_c

                if bucketIdx == 0:
                    continue
                else:
                    buckets[bucketIdx] += p

            acc = 0
            runningSum = 0
            for j in range(two_c-1, 0, -1):
                runningSum += buckets[j]
                acc += runningSum
                buckets[j] = 0

            windowSums[windowIdx] = acc

        result[0] = 0
        for window in windowSums[::-1]:
            for _ in range(nbuckets):
                result[0] += result[0]
            result[0] += window    

    # Allocate the result array, which is a one element array (https://numba.readthedocs.io/en/stable/cuda/kernels.html#kernel-declaration)
    result = cuda.device_array((1,), np.uint32)

    # Just run it with 1 thread for now.
    pippengers[1, 1](scalars, points, result)
    return result[0]
