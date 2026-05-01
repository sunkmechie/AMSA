import amsa

alg = amsa.Algebra.cga3d()

a = alg.point([1.0, 2.0, 3.0])
b = alg.point([4.0, 2.0, 3.0])

d2 = alg.distance_squared(a, b)

print(d2)  # 9.0
