import numpy as np
from ecpy.curves import Curve, Point

def generate_points(n=100):
    '''
    Randomly generate n scalars and curve points on secp256k1
    Curve points are random linear combinations of P and Q
    '''
    cv = Curve.get_curve('secp256k1')
    px = 0x65d5b8bf9ab1801c9f168d4815994ad35f1dcb6ae6c7a1a303966b677b813b00
    py = 0xe6b865e529b8ecbf71cf966e900477d49ced5846d7662dd2dd11ccd55c0aff7f
    qx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    qy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    P  = Point(px, py, cv)
    Q  = Point(qx, qy, cv)

    scalars = np.random.randint(-10000, 10000, size=(n,))
    xs = np.random.randint(-100, 100, size=(n,2))
    points = [(a*P + b*Q) for a,b in xs]
    return scalars, points

def generate_scalar_vectors(n=100):
    '''
    Randomly generate 2 vectors of n scalars.
    '''
    scalars1 = np.random.randint(-10000, 10000, size=(n,))
    scalars2 = np.random.randint(-10000, 10000, size=(n,))
    return scalars1, scalars2

def manual_msm(es, gs):
    '''
    es: list of ints
    gs: list of elliptic curve points
    '''
    acc = Point.infinity()
    for e, g in zip(es, gs):
        acc += e*g

    return acc
