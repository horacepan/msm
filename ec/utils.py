import numpy as np
from .ecp import ECP

def _generate_random(P, Q, n):
    '''
    Generate n random linear combinations of elliptic curve points P and Q
    '''
    xs = np.random.randint(-100, 100, size=(n, 2))
    gs = [a*P + b*Q for a, b in xs]
    return gs

def gen_secp256k1(n):
    a = 0
    b = 7
    q = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f

    x1 = 0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
    y1 = 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8
    x2 = 0x1d3b328dae52c976dfb103c4fccef6f9a09bd06431c9dbaaf295a9a487b41d8a
    y2 = 0x784cefd60c8299a85f6e9ab15bad8f4453b16f75b9b6d504a2fd909205d2a2f1
    P = ECP(x1, y1, a, b, q)
    Q = ECP(x2, y2, a, b, q)
    return _generate_random(P, Q, n)

def gen_32bit_pts(n):
    '''
    Generate random points from a 32 bit curve
    Ref: https://crypto.stackexchange.com/questions/45053/32-bit-or-16-bits-elliptic-curves
    '''
    a = 1456400922
    b = 2005615003
    q = 2**31 - 1

    x1 = 1989721171
    y1 = 1981657285
    x2 = 99655887
    y2 = 1649525007
    P = ECP(x1, y1, a, b, q)
    Q = ECP(x2, y2, a, b, q)
    return _generate_random(P, Q, n)
