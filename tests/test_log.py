import numpy as np
import pytest

import amsa


def test_generic_log_inverts_simple_circular_bivector_exp() -> None:
    alg = amsa.Algebra.vga2d()
    generator = alg.bivector([0.25])
    rotor = amsa.exp(generator)

    recovered = amsa.log(rotor)

    assert recovered.layout.grades == (2,)
    assert np.allclose(recovered.values, generator.values)


def test_generic_log_rejects_negative_real_scalar_branch() -> None:
    alg = amsa.Algebra.vga2d()

    with pytest.raises(ValueError, match="negative scalars"):
        amsa.log(alg.scalar(-1.0))


def test_generic_log_rejects_non_simple_bivector_square() -> None:
    alg = amsa.Algebra(amsa.AlgebraSpec.from_pqr(4))
    rotor_like = alg.scalar(1.0) + alg.multivector({"e12": 0.1, "e34": 0.2})

    with pytest.raises(ValueError, match="must be scalar-valued"):
        amsa.log(rotor_like)
