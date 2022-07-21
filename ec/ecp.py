from .ecp_functional import add, sub, mul, on_curve

class ECP:
    def __init__(self, x, y, a, b, q, is_inf=False):
        '''
        Elliptic curve point in affine form with the curve expressed
        in Weierstrass form.
        '''
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.q = q
        self.is_inf = is_inf

    def __add__(self, Q):
        if self.is_inf:
            return Q
        if Q.is_inf:
            return self

        px, py = self.x, self.y
        qx, qy = Q.x, Q.y
        a, b = self.a, self.b
        q = self.q
        x, y, is_inf = add(px, py, qx, qy, a, b, q)
        return ECP(x, y, a, b, q, is_inf)

    def __mul__(self, k):
        if self.is_inf:
            return self

        x, y = self.x, self.y
        a, b = self.a, self.b
        q = self.q
        x, y, is_inf = mul(k, x, y, a, b, q)
        return ECP(x, y, a, b, q, is_inf)

    def __rmul__(self, k):
        return self.__mul__(k)

    def __lmul__(self, k):
        return self.__mul__(k)

    def __eq__(self, Q):
        if self.is_inf and Q.is_inf:
            return True

        return self.x == Q.x and self.y == Q.y and self.is_inf == Q.is_inf

    def on_curve(self):
        return on_curve(self.x, self.y, self.a, self.b, self.q)


def infinity():
    return ECP(0, 0, 0, 0, 0, True)
