'''
Functional implementation of operations on elliptic curves in
Weierstrass form.

Points (x,y) on an elliptic curve represented by the equation y^2 = x^3 + ax + b mod q
can be represented by a 5 ints (x, y, z, a, b, q).

Adapted from ECPy's curve module:
https://ec-python.readthedocs.io/en/latest/
'''
def aff2jac(x, y) -> tuple[int, int, int]:
    '''
    Convert from affine coordinates to jacobian
    '''
    return (x, y, 1)

def jac2aff(x, y, z, q) -> tuple[int, int]:
    '''
    Convert from jacobian coordinates to affine
    '''
    invz = pow(z, q-2, q) # z^{p-1} = 1 mod p, so z^{p-2} * z = 1 mod p
    sqinvz = (invz*invz)%q
    x = (x*sqinvz)%q
    y = (y*sqinvz*invz)%q
    return (x, y)

def add(px, py, qx, qy, a, b, q) -> tuple[int, int, bool]:
    '''
    P = (Px, Py), Q = (Qx, Qy) on the curve: y^2 = x^3 + ax + b mod q
    Returns:
    - the x, y coordinates
    - a bool indicating whether the point is the infinity point.
    '''
    if px == qx and py == qy:
        Px, Py, Pz = aff2jac(px,py)
        x, y, z = dbl_jac(Px, Py, Pz, q, a)
    else:
        Px, Py, Pz = aff2jac(px, py)
        Qx, Qy, Qz = aff2jac(qx, qy)
        x, y, z = add_jac(Px, Py, Pz, Qx, Qy, Qz, q)

    if z:
        x, y = jac2aff(x, y, z, q)
        return x, y, False
    else:
        return 0, 0, True

def sub(px, py, qx, qy, a, b, q) -> tuple[int, int, bool]:
    return add(px, py, qx, -qy, a, b, q)

def dbl_jac(X1, Y1, Z1, q, a) -> tuple[int, int, int]:
    XX   = (X1*X1)%q
    YY   = (Y1*Y1)%q
    YYYY = (YY*YY)%q
    ZZ   = (Z1*Z1)%q
    S    = (2*((X1+YY)*(X1+YY)-XX-YYYY))%q
    M    = (3*XX+a*ZZ*ZZ)%q
    T    = (M*M-2*S)%q
    X3   = (T)%q
    Y3   = (M*(S-T)-8*YYYY)%q
    Z3   = ((Y1+Z1)*(Y1+Z1)-YY-ZZ)%q
    return X3,Y3,Z3

def add_jac(X1,Y1,Z1, X2,Y2,Z2, q) -> tuple[int, int, int]:
    Z1Z1 = (Z1*Z1)%q
    Z2Z2 = (Z2*Z2)%q
    U1   = (X1*Z2Z2)%q
    U2   = (X2*Z1Z1)%q
    S1   = (Y1*Z2*Z2Z2)%q
    S2   = (Y2*Z1*Z1Z1)%q
    H    = (U2-U1)%q
    I    = ((2*H)*(2*H))%q
    J    = (H*I)%q
    r    = (2*(S2-S1))%q
    V    = (U1*I)%q
    X3   = (r*r-J-2*V)%q
    Y3   = (r*(V-X3)-2*S1*J)%q
    Z3   = (((Z1+Z2)*(Z1+Z2)-Z1Z1-Z2Z2)*H)%q
    return X3,Y3,Z3

def mul(k, px, py, a, b, q) -> [int, int, bool]:
    '''
    Scalar multiplication
    '''
    kbin = bin(k)[2:]
    x1, y1, z1 = aff2jac(px, py)
    x2, y2, z2 = dbl_jac(x1, y1, z1, q, a)

    for i in range(1, len(kbin)):
        if kbin[i] == '1':
            x1,y1,z1 = add_jac(x2,y2,z2,x1,y1,z1, q)
            x2,y2,z2 = dbl_jac(x2,y2,z2,q,a)
        else:
            x2,y2,z2 = add_jac(x1,y1,z1,x2,y2,z2,q)
            x1,y1,z1 = dbl_jac(x1,y1,z1,q,a)

    if z1:
        x, y = jac2aff(x1, y1, z1, q)
        return x, y, False
    else:
        return 0, 0, True

def on_curve(x, y, a, b, q) -> bool:
    rhs = (pow(x, 3, q) + a*x + b) % q
    y2 = pow(y, 2, q)
    return (y2 == rhs)






import numba.cuda as cuda
import math

@cuda.jit
def gpu_add(px, py, qx, qy, a, b, q) -> tuple[int, int, bool]:
    '''
    P = (Px, Py), Q = (Qx, Qy) on the curve: y^2 = x^3 + ax + b mod q
    Returns:
    - the x, y coordinates
    - a bool indicating whether the point is the infinity point.
    '''
    if px == qx and py == qy:
        Px, Py, Pz = gpu_aff2jac(px,py)
        x, y, z = gpu_dbl_jac(Px, Py, Pz, q, a)
    else:
        Px, Py, Pz = gpu_aff2jac(px, py)
        Qx, Qy, Qz = gpu_aff2jac(qx, qy)
        x, y, z = gpu_add_jac(Px, Py, Pz, Qx, Qy, Qz, q)

    if z:
        x, y = gpu_jac2aff(x, y, z, q)
        return x, y, False
    else:
        return 0, 0, True

@cuda.jit
def gpu_aff2jac(x, y) -> tuple[int, int, int]:
    '''
    Convert from affine coordinates to jacobian
    '''
    return (x, y, 1)

@cuda.jit
def gpu_dbl_jac(X1, Y1, Z1, q, a) -> tuple[int, int, int]:
    XX   = (X1*X1)%q
    YY   = (Y1*Y1)%q
    YYYY = (YY*YY)%q
    ZZ   = (Z1*Z1)%q
    S    = (2*((X1+YY)*(X1+YY)-XX-YYYY))%q
    M    = (3*XX+a*ZZ*ZZ)%q
    T    = (M*M-2*S)%q
    X3   = (T)%q
    Y3   = (M*(S-T)-8*YYYY)%q
    Z3   = ((Y1+Z1)*(Y1+Z1)-YY-ZZ)%q
    return X3,Y3,Z3

@cuda.jit
def gpu_add_jac(X1,Y1,Z1, X2,Y2,Z2, q) -> tuple[int, int, int]:
    Z1Z1 = (Z1*Z1)%q
    Z2Z2 = (Z2*Z2)%q
    U1   = (X1*Z2Z2)%q
    U2   = (X2*Z1Z1)%q
    S1   = (Y1*Z2*Z2Z2)%q
    S2   = (Y2*Z1*Z1Z1)%q
    H    = (U2-U1)%q
    I    = ((2*H)*(2*H))%q
    J    = (H*I)%q
    r    = (2*(S2-S1))%q
    V    = (U1*I)%q
    X3   = (r*r-J-2*V)%q
    Y3   = (r*(V-X3)-2*S1*J)%q
    Z3   = (((Z1+Z2)*(Z1+Z2)-Z1Z1-Z2Z2)*H)%q
    return X3,Y3,Z3

@cuda.jit
def gpu_jac2aff(x, y, z, q) -> tuple[int, int]:
    '''
    Convert from jacobian coordinates to affine
    '''
    # z^{p-1} = 1 mod p, so z^{p-2} * z = 1 mod p
    invz = math.pow(z, q-2)
    invz = invz%q

    sqinvz = (invz*invz)%q
    x = (x*sqinvz)%q
    y = (y*sqinvz*invz)%q
    return (x, y)
