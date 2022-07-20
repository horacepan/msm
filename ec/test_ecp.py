import numpy as np
from ecp import ECP

def test_secp256k1():
    a = 0
    b = 7
    q = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
    gx = 0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
    gy = 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8

    p1 = ECP(gx, gy, a, b, q)
    p2 = p1 + p1
    p4 = p2 + p2
    p8 = p4 + p4
    p13 = p8 + p4 + p2 - p1

    assert (p13 == (13 * p1))
    print('Okay!')

def test():
    test_secp256k1()

if __name__ == '__main__':
    test()
