from ecpy.curves import Curve,Point

def pip_msm(es, gs, num_bits, c):
    '''
    es: list of scalars
    gs: list of EC Points
    num_bits: int, number of bits of scalars
    c: bit window to use for Pippengers
    Returns: \sum_i e_i g_i
    '''
    nbucket = num_bits // c
    two_c = int(2**c)
    cidx = 0
    window_sums = []

    while cidx <= num_bits:
        # Point.inf + g = g + Point.inf = g for all g \in curve
        bucket = [Point.infinity()] * (two_c)
        for e, g in zip(es, gs):
            b = (e >> cidx) % two_c # bit shift, grab last c bits
            if b == 0:
                continue
            else:
                bucket[b] += g

        acc = Point.infinity()
        running_sum = Point.infinity()
        for j in range(len(bucket) - 1, 0, -1):
            running_sum += bucket[j]
            acc += running_sum

        cidx += c
        window_sums.append(acc)

    total = Point.infinity()
    for window in window_sums[::-1]:
        for _ in range(c):
            total += total
        total += window

    return total
