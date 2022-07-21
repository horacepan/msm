from dadd import double_add
from utils import generate_points
import numpy as np

def test(n=10):
    es, gs = generate_points('secp256k1', n)
    es = np.abs(es) # TODO: handle negative case

    for e, g in zip(es, gs):
        true_eg = e * g
        man_eg = double_add(e, g)

        assert(true_eg == man_eg)
    print("test passed")

if __name__ == '__main__':
    test()
