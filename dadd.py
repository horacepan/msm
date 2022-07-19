def to_binary(s):
    bits = []
    while s > 0:
        b = s % 2
        bits.append(b)
        s = s >> 1

    bits = bits[::-1]
    return bits

def double_add(s, g):
    if s < 0:
        return (-1) * double_add(abs(s), g)

    sbin = to_binary(s)
    out = g

    for i in sbin[1:]:
        out += out
        if i:
            out += g
    return out
