import numpy as np
import ec.utils
import ec.ecp

def generate_points(curve_name, n=100):
    '''
    Randomly generate n scalars and curve points
    Curve points are random linear combinations of P and Q
    '''
    if curve_name == 'secp256k1':
        points = ec.utils.gen_secp256k1(n)
    elif curve_name == '32bit':
        points = ec.utils.gen_32bit_pts(n)
    else:
        raise Exception("Invalid curve_name.  Valid choses are 'secp256k1' and '32bit'")

    scalars = np.random.randint(-10000, 10000, size=(n,))
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
    acc = ec.ecp.infinity()
    for e, g in zip(es, gs):
        acc += e*g

    return acc
