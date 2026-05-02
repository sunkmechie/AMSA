import json
import math

import numpy as np

import amsa.robo as robo
from amsa import Algebra
from tests._utils import assert_allclose


def test_planar_two_link_ik_reaches_target() -> None:
    q1, q2 = robo.ik((1.0, 1.0), (1.0, 1.0), solver="planar_two_link")

    x = math.cos(q1) + math.cos(q1 + q2)
    y = math.sin(q1) + math.sin(q1 + q2)
    assert math.isclose(x, 1.0, abs_tol=1e-12)
    assert math.isclose(y, 1.0, abs_tol=1e-12)


def test_importurdf_and_crobot_roundtrip_shape(tmp_path) -> None:
    path = tmp_path / "arm.urdf"
    path.write_text(
        """
        <robot name="two_link">
          <link name="base"/>
          <link name="tip"/>
          <joint name="joint1" type="revolute">
            <parent link="base"/>
            <child link="tip"/>
            <origin xyz="1 0 0" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
          </joint>
        </robot>
        """,
        encoding="utf-8",
    )

    model = robo.importurdf(path)
    data = robo.dump_crobot(model)
    crobot = tmp_path / "arm.crobot"
    crobot.write_text(json.dumps(data), encoding="utf-8")

    loaded = robo.load_crobot(crobot)
    assert loaded.name == "two_link"
    assert loaded.joints[0].axis == (0.0, 0.0, 1.0)
    assert data["joints"][0]["motion"] == "bivector-generator"


# -- DH-parameterized FK tests -------------------------------------------------


def _extract(results, i):
    return Algebra.cga3d().extract_point(results[i][1])


def test_fk_two_link_zero_angles() -> None:
    alg = Algebra.cga3d()
    # dh_params: (α, a, d, θ) — zero twist, link along x, θ varies
    results = robo.fk(alg, [(0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)])
    assert_allclose(_extract(results, 0), [1.0, 0.0, 0.0], atol=1e-15)
    assert_allclose(_extract(results, 1), [2.0, 0.0, 0.0], atol=1e-15)


def test_fk_two_link_half_pi() -> None:
    alg = Algebra.cga3d()
    results = robo.fk(alg, [(0.0, 1.0, 0.0, math.pi / 2), (0.0, 1.0, 0.0, 0.0)])
    assert_allclose(_extract(results, 0), [0.0, 1.0, 0.0], atol=1e-15)
    assert_allclose(_extract(results, 1), [0.0, 2.0, 0.0], atol=1e-15)


def test_fk_two_link_full() -> None:
    alg = Algebra.cga3d()
    t1, t2 = math.pi / 4, math.pi / 4
    results = robo.fk(alg, [(0.0, 1.0, 0.0, t1), (0.0, 1.0, 0.0, t2)])
    p1 = _extract(results, 0)[:2]
    p2 = _extract(results, 1)[:2]
    expected_p1 = np.array([math.cos(t1), math.sin(t1)])
    expected_p2 = np.array(
        [math.cos(t1) + math.cos(t1 + t2), math.sin(t1) + math.sin(t1 + t2)]
    )
    assert_allclose(p1, expected_p1, atol=1e-15)
    assert_allclose(p2, expected_p2, atol=1e-15)


def test_fk_three_link() -> None:
    alg = Algebra.cga3d()
    results = robo.fk(
        alg,
        [
            (0.0, 1.0, 0.0, math.pi / 6),
            (0.0, 0.5, 0.0, math.pi / 3),
            (0.0, 0.3, 0.0, math.pi / 4),
        ],
    )
    p3 = _extract(results, 2)
    assert p3.shape == (3,)


def test_fk_motor_is_even() -> None:
    alg = Algebra.cga3d()
    results = robo.fk(alg, [(0.0, 1.0, 0.0, 0.5), (0.0, 0.8, 0.0, -0.3)])
    for motor, _ in results:
        assert set(motor.grades).issubset({0, 2, 4})


def test_fk_twisted_joint_alpha_pi_over_2() -> None:
    alg = Algebra.cga3d()
    # α=π/2 twists the next joint axis from z to y
    results = robo.fk(alg, [(math.pi / 2, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, math.pi / 2)])
    # After α twist, the second link goes along y (was z before twist, then x translation)
    # At θ1=0, α twist makes the next frame's z point along old -y
    p1 = _extract(results, 0)
    assert_allclose(p1, [0.0, 0.0, 1.0], atol=1e-10)
    assert _extract(results, 1).shape == (3,)


def test_fk_prismatic_joint() -> None:
    alg = Algebra.cga3d()
    results = robo.fk(
        alg,
        [(0.0, 0.0, 2.0, 0.0), (0.0, 1.0, 0.0, 0.0)],
        joint_types=["prismatic", "revolute"],
    )
    assert_allclose(_extract(results, 0), [0.0, 0.0, 2.0], atol=1e-15)
    assert_allclose(_extract(results, 1), [1.0, 0.0, 2.0], atol=1e-15)


def test_fk_five_dof() -> None:
    alg = Algebra.cga3d()
    results = robo.fk(
        alg,
        [
            (0.0, 0.0, 0.5, 0.3),
            (0.0, 0.8, 0.0, -0.2),
            (math.pi / 2, 0.0, 0.3, 0.1),
            (0.0, 0.6, 0.0, 0.0),
            (0.0, 0.0, 0.2, 0.4),
        ],
    )
    assert len(results) == 5
    p5 = _extract(results, 4)
    assert p5.shape == (3,)
