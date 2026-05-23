import numpy as np

import amsa
from amsa import Algebra
from tests._utils import assert_allclose


def test_directional_derivative_geometric_product_uses_product_rule() -> None:
    alg = Algebra.vga2d()
    layout = alg.grade_layout(1)
    point = alg.multivector([2.0, -1.0], layout=layout)
    seed = alg.multivector([0.5, 3.0], layout=layout)

    tangent = amsa.directional_derivative(lambda x: x * x, point, seed)

    expected = seed * point + point * seed
    assert_allclose(tangent.values, expected.values)


def test_forward_grad_scalar_product_matches_closed_form() -> None:
    alg = Algebra.vga3d()
    layout = alg.grade_layout(1)
    point = alg.multivector([0.5, -1.5, 2.0], layout=layout)
    weights = alg.multivector([2.0, -3.0, 0.25], layout=layout)

    gradient = amsa.forward_grad(lambda x: x.scalar_product(weights), point)

    assert_allclose(gradient, np.array([2.0, -3.0, 0.25]))


def test_forward_grad_norm_squared_matches_two_x_for_euclidean_vector() -> None:
    alg = Algebra.vga3d()
    layout = alg.grade_layout(1)
    point = alg.multivector([0.5, -1.5, 2.0], layout=layout)

    gradient = amsa.forward_grad(lambda x: x.norm_squared(), point)

    assert_allclose(gradient, np.array([1.0, -3.0, 4.0]))
