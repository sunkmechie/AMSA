import _bootstrap

from amsa import Algebra
from tests._utils import assert_mv_allclose
import numpy as np

print("\n=== Challenge 6: Even/Odd Decomposition ===")

alg = Algebra.vga3d()

mv = alg.multivector({
    "e1":1.0,
    "e2":-0.5,
    "e12":2.0,
    "e123":3.0
})

even = mv.grade(0,2)
odd = mv.grade(1,3)

recomposed = even + odd

print("original:", mv.values)
print("even:", even.values)
print("odd:", odd.values)

assert_mv_allclose(mv, recomposed)

print("Even/Odd decomposition verified.")
