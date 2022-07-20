from ecpy.curves import WeierstrassCurve
from ecpy.curve_defs import WEIERSTRASS

'''
Parameters for bn254 curve
Ref: https://neuromancer.sk/std/bn/bn254
Param dict must match the format specified by:
https://ec-python.readthedocs.io/en/latest/_modules/ecpy/curves.html#WeierstrassCurve
'''
bn254_params = {
    'name': 'bn254',
    'type': WEIERSTRASS,
    'size': 254,
    'a': 0,
    'b': 2,
    'field': 0x2523648240000001BA344D80000000086121000000000013A700000000000013,
    'generator': (0x2523648240000001BA344D80000000086121000000000013A700000000000012, 0x0000000000000000000000000000000000000000000000000000000000000001),
    'order': 0x2523648240000001BA344D8000000007FF9F800000000010A10000000000000D,
    'cofactor': 1
}
bn254 = WeierstrassCurve(bn254_params)
