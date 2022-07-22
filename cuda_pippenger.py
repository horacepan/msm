import numba.cuda as cuda
import numba
import numpy as np
import math
import ec.ecp
import ec.ecp_functional as ecpf

def cuda_pip_msm(scalars, points, num_bits, c):
    if len(scalars) == 0:
        return ec.ecp.infinity()

    nbuckets = math.ceil(num_bits/c)
    two_c = int(2**c)

    @cuda.jit
    def pippengers(scalars, xyCoords, a, b, q, result):
        # Allocate local memory for the window buckets and
        # for the window sums
        buckets = numba.cuda.local.array((two_c, 2), np.uint64)
        windowSums = numba.cuda.local.array((nbuckets, 2), np.uint64)
        acc = numba.cuda.local.array((2,), np.uint64)
        runningSum = numba.cuda.local.array((2,), np.uint64)

        for windowIdx in range(nbuckets):
            rightShift = c * windowIdx

            for s, p in zip(scalars, xyCoords):
                bucketIdx = (s >> rightShift) % two_c

                if bucketIdx == 0:
                    continue
                else:
                    bucketVal = buckets[bucketIdx]
                    bucketVal[0], bucketVal[1], _ = ecpf.gpu_add(bucketVal[0], bucketVal[1], p[0], p[1], a, b, q)
                    buckets[bucketIdx][0] = bucketVal[0]
                    buckets[bucketIdx][1] = bucketVal[1]

            acc[0] = 0
            acc[1] = 0
            runningSum[0] = 0
            runningSum[1] = 0
            for j in range(two_c-1, 0, -1):
                runningSum[0], runningSum[1], _ = ecpf.gpu_add(runningSum[0], runningSum[1], buckets[j][0], buckets[j][1], a, b, q)
                acc[0], acc[1], _ = ecpf.gpu_add(acc[0], acc[1], runningSum[0], runningSum[1], a, b, q)
                buckets[j][0] = 0
                buckets[j][1] = 0

            windowSums[windowIdx][0] = acc[0]
            windowSums[windowIdx][1] = acc[1]

        for window in windowSums[::-1]:
            for _ in range(c):
                result[0], result[1], _ = ecpf.gpu_add(result[0], result[1], result[0], result[1], a, b, q)
            result[0], result[1], _ = ecpf.gpu_add(result[0], result[1], window[0], window[1], a, b, q)

    # Convert the points to an numpy array of x and y coordinates
    xyCoords = np.asarray([[p.x, p.y] for p in points], dtype=np.uint32)
    d_xyCoords = cuda.to_device(xyCoords)
    d_scalars = cuda.to_device(scalars)

    # Allocate the result array, which is a one element array (https://numba.readthedocs.io/en/stable/cuda/kernels.html#kernel-declaration)
    result = cuda.device_array((2,), np.uint64)

    # Just run it with 1 thread for now.
    pippengers[1, 1](d_scalars, d_xyCoords, points[0].a, points[0].b, points[0].q, result)
    result.copy_to_host()

    return result
