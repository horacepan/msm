import ec.ecp

def pip_msm(es, gs, num_bits, c):
    '''
    es: list of scalars
    gs: list of EC Points
    num_bits: int, number of bits of scalars
    c: bit window to use for Pippengers
    Returns: \sum_i e_i g_i
    '''
    nbuckets = num_bits // c
    two_c = int(2**c)
    window_sums = [ec.ecp.infinity()] * (nbuckets + 1)

    for widx in range(nbuckets + 1):
        # Point.inf + g = g + Point.inf = g for all g \in curve
        right_shift = c * widx
        bucket = [ec.ecp.infinity()] * (two_c)
        for e, g in zip(es, gs):
            b = (e >> right_shift) % two_c # bit shift, grab last c bits
            if b == 0:
                continue
            else:
                bucket[b] += g

        acc = ec.ecp.infinity()
        running_sum = ec.ecp.infinity()
        for j in range(len(bucket) - 1, 0, -1):
            running_sum += bucket[j]
            acc += running_sum

        window_sums[widx] = acc

    total = ec.ecp.infinity()
    for window in window_sums[::-1]:
        for _ in range(c):
            total += total
        total += window

    return total
