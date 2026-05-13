import json
import math

import numpy as np
import pytest

import amsa.robo as robo
from amsa import Algebra
from tests._utils import assert_allclose


def test_planar_two_link_ik_reaches_target() -> None:
    q1, q2 = robo.ik((1.0, 1.0), (1.0, 1.0), solver="planar_two_link")

    x = math.cos(q1) + math.cos(q1 + q2)
    y = math.sin(q1) + math.sin(q1 + q2)
    assert math.isclose(x, 1.0, abs_tol=1e-12)
    assert math.isclose(y, 1.0, abs_tol=1e-12)


def test_cga_sphere_sphere_returns_intersection_circle() -> None:
    alg = Algebra.cga3d()
    s1 = alg.sphere([0.0, 0.0, 0.0], 1.0)
    s2 = alg.sphere([1.0, 0.0, 0.0], 1.0)

    circle = robo.sphere_sphere(s1, s2)
    p = alg.point([0.5, math.sqrt(0.75), 0.0])

    assert 3 in circle.grades
    assert_allclose((p ^ circle).values, np.zeros_like((p ^ circle).values), atol=1e-12)
    assert_allclose(robo.ik(s1, s2, solver="cga_sphere_sphere").values, circle.values)


def test_cga_line_plane_returns_intersection_point() -> None:
    alg = Algebra.cga3d()
    line = alg.line_through_points(
        alg.point([0.0, 0.0, 0.0]),
        alg.point([1.0, 0.0, 0.0]),
    )
    plane = alg.plane([1.0, 0.0, 0.0], 2.0)

    point = robo.line_plane(line, plane)

    assert_allclose(alg.extract_point(point), [2.0, 0.0, 0.0], atol=1e-12)


def test_cga_point_circle_projection() -> None:
    alg = Algebra.cga3d()
    circle = alg.circle_through_points(
        alg.point([1.0, 0.0, 0.0]),
        alg.point([0.0, 1.0, 0.0]),
        alg.point([-1.0, 0.0, 0.0]),
    )
    point = alg.point([2.0, 2.0, 3.0])

    projected = robo.point_circle_projection(point, circle)

    coords = alg.extract_point(projected)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    assert_allclose(coords, [inv_sqrt2, inv_sqrt2, 0.0], atol=1e-12)
    incidence = projected ^ circle
    assert_allclose(incidence.values, np.zeros_like(incidence.values), atol=1e-12)
    assert_allclose(
        robo.ik(point, circle, solver="cga_point_circle").values,
        projected.values,
    )


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


def test_crobot_roundtrip_preserves_executable_offsets(tmp_path) -> None:
    model = robo.model_from_dh(
        [(math.pi / 2, 0.5, 0.25, 0.0)],
        name="offset_chain",
    )
    data = robo.dump_crobot(model)
    path = tmp_path / "offset_chain.crobot"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = robo.load_crobot(path)

    assert loaded.joints[0].origin_xyz == (0.0, 0.0, 0.25)
    assert loaded.joints[0].child_offset_xyz == (0.5, 0.0, 0.0)
    assert loaded.joints[0].child_offset_rpy == (math.pi / 2, 0.0, 0.0)


# -- DH-parameterized FK tests -------------------------------------------------


def test_fk_two_link_zero_angles() -> None:
    alg = Algebra.cga3d()
    results = robo.fk(alg, [(0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)])
    assert_allclose(results[0]["position"], [1.0, 0.0, 0.0], atol=1e-15)
    assert_allclose(results[1]["position"], [2.0, 0.0, 0.0], atol=1e-15)


def test_model_from_dh_fk_matches_dh_fk() -> None:
    alg = Algebra.cga3d()
    dh_home = [
        (0.0, 1.0, 0.2, 0.0),
        (math.pi / 2, 0.5, 0.0, 0.0),
        (0.0, 0.3, 0.1, 0.0),
    ]
    joint_values = np.array([0.4, -0.2, 0.7])
    model = robo.model_from_dh(dh_home)
    dh_active = [
        (alpha, a, d, float(theta))
        for (alpha, a, d, _), theta in zip(dh_home, joint_values, strict=True)
    ]

    from_dh = robo.fk(alg, dh_active)
    from_model = robo.fk_model(alg, model, joint_values)

    assert len(from_model) == len(from_dh)
    assert_allclose(from_model[-1]["position"], from_dh[-1]["position"], atol=1e-12)
    assert_allclose(
        robo.motor_to_matrix(from_model[-1]["motor"], alg),
        robo.motor_to_matrix(from_dh[-1]["motor"], alg),
        atol=1e-12,
    )


def test_model_fk_prismatic_matches_dh_fk() -> None:
    alg = Algebra.cga3d()
    dh_home = [(0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)]
    joint_types = ["prismatic", "revolute"]
    model = robo.model_from_dh(dh_home, joint_types=joint_types)
    joint_values = np.array([0.5, 0.3])
    dh_active = [(0.0, 0.0, 0.5, 0.0), (0.0, 1.0, 0.0, 0.3)]

    from_dh = robo.fk(alg, dh_active, joint_types=joint_types)
    from_model = robo.fk_model(alg, model, joint_values)

    assert_allclose(from_model[-1]["position"], from_dh[-1]["position"], atol=1e-12)


def test_fk_two_link_half_pi() -> None:
    alg = Algebra.cga3d()
    results = robo.fk(alg, [(0.0, 1.0, 0.0, math.pi / 2), (0.0, 1.0, 0.0, 0.0)])
    assert_allclose(results[0]["position"], [0.0, 1.0, 0.0], atol=1e-15)
    assert_allclose(results[1]["position"], [0.0, 2.0, 0.0], atol=1e-15)


def test_fk_two_link_full() -> None:
    alg = Algebra.cga3d()
    t1, t2 = math.pi / 4, math.pi / 4
    results = robo.fk(alg, [(0.0, 1.0, 0.0, t1), (0.0, 1.0, 0.0, t2)])
    p1 = results[0]["position"][:2]
    p2 = results[1]["position"][:2]
    expected_p1 = np.array([math.cos(t1), math.sin(t1)])
    expected_p2 = np.array([math.cos(t1) + math.cos(t1 + t2), math.sin(t1) + math.sin(t1 + t2)])
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
    p3 = results[2]["position"]
    assert p3.shape == (3,)


def test_fk_motor_is_even() -> None:
    alg = Algebra.cga3d()
    results = robo.fk(alg, [(0.0, 1.0, 0.0, 0.5), (0.0, 0.8, 0.0, -0.3)])
    for r in results:
        assert set(r["motor"].grades).issubset({0, 2, 4})


def test_fk_twisted_joint_alpha_pi_over_2() -> None:
    alg = Algebra.cga3d()
    results = robo.fk(alg, [(math.pi / 2, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, math.pi / 2)])
    p1 = results[0]["position"]
    assert_allclose(p1, [0.0, 0.0, 1.0], atol=1e-10)
    assert results[1]["position"].shape == (3,)


def test_fk_prismatic_joint() -> None:
    alg = Algebra.cga3d()
    results = robo.fk(
        alg,
        [(0.0, 0.0, 2.0, 0.0), (0.0, 1.0, 0.0, 0.0)],
        joint_types=["prismatic", "revolute"],
    )
    assert_allclose(results[0]["position"], [0.0, 0.0, 2.0], atol=1e-15)
    assert_allclose(results[1]["position"], [1.0, 0.0, 2.0], atol=1e-15)


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
    p5 = results[4]["position"]
    assert p5.shape == (3,)


def test_fk_orientation_is_unit_quaternion() -> None:
    alg = Algebra.cga3d()
    results = robo.fk(alg, [(0.0, 1.0, 0.0, 0.5), (0.0, 0.8, 0.0, -0.3)])
    for r in results:
        q = r["orientation"]
        assert_allclose(np.linalg.norm(q), 1.0, atol=1e-15)


def test_fk_home_orientation_is_identity() -> None:
    alg = Algebra.cga3d()
    results = robo.fk(alg, [(0.0, 1.0, 0.0, 0.0)])
    q = results[0]["orientation"]
    assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-15)


def test_fk_rejects_non_cga3d() -> None:
    with pytest.raises(ValueError, match="cga3d"):
        robo.fk(Algebra.cga2d(), [(0.0, 1.0, 0.0, 0.0)])


def test_fk_rejects_bad_joint_types() -> None:
    alg = Algebra.cga3d()
    with pytest.raises(ValueError, match="Expected 2 joint types"):
        robo.fk(
            alg,
            [(0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)],
            joint_types=["revolute"],
        )
    with pytest.raises(ValueError, match="Unsupported joint type"):
        robo.fk(alg, [(0.0, 1.0, 0.0, 0.0)], joint_types=["helical"])


# -- DLS IK tests -------------------------------------------------------------


def test_ik_dls_two_link_zero_angle_target() -> None:
    alg = Algebra.cga3d()
    dh = [(0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)]
    target = robo.fk(alg, dh)[-1]["motor"]
    result = robo.ik_dls(alg, dh, target, position_tolerance=1e-8, orientation_tolerance=1e-8)
    assert result.success
    assert result.iterations <= 5
    assert_allclose(result.joint_angles, [0.0, 0.0], atol=1e-6)


def test_ik_dls_zero_iterations_returns_failure_result() -> None:
    alg = Algebra.cga3d()
    dh = [(0.0, 1.0, 0.0, 0.0)]
    target = robo.fk(alg, dh)[-1]["motor"]
    result = robo.ik_dls(alg, dh, target, max_iterations=0)
    assert not result.success
    assert result.iterations == 0
    assert result.position is not None


def test_ik_dls_rejects_invalid_shapes_and_algebras() -> None:
    alg = Algebra.cga3d()
    dh = [(0.0, 1.0, 0.0, 0.0)]
    target = robo.fk(alg, dh)[-1]["motor"]
    with pytest.raises(ValueError, match="initial joint"):
        robo.ik_dls(alg, dh, target, initial_angles=np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="joint limits"):
        robo.ik_dls(alg, dh, target, joint_limits=[])
    with pytest.raises(ValueError, match="provided algebra"):
        robo.ik_dls(alg, dh, Algebra.cga2d().translate([1.0, 0.0]))


def test_ik_dls_two_link_vs_analytic() -> None:
    alg = Algebra.cga3d()
    dh = [(0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)]

    q1_target, q2_target = 0.4, 0.6
    target = robo.fk(alg, [(0.0, 1.0, 0.0, q1_target), (0.0, 1.0, 0.0, q2_target)])[-1]["motor"]

    result = robo.ik_dls(alg, dh, target, position_tolerance=1e-8, orientation_tolerance=1e-8)
    assert result.success
    assert_allclose(result.joint_angles, [q1_target, q2_target], atol=1e-5)


def test_ik_dls_two_link_target_position() -> None:
    alg = Algebra.cga3d()
    dh = [(0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)]
    fk_params = [(0.0, 1.0, 0.0, math.pi / 6), (0.0, 1.0, 0.0, math.pi / 3)]
    target_motor = robo.fk(alg, fk_params)[-1]["motor"]
    result = robo.ik_dls(alg, dh, target_motor)
    assert result.success
    q0, q1 = result.joint_angles[0], result.joint_angles[1]
    fk_check = robo.fk(alg, [(0.0, 1.0, 0.0, q0), (0.0, 1.0, 0.0, q1)])
    assert_allclose(fk_check[-1]["position"], result.position, atol=1e-6)


def test_ik_dls_three_link_converges() -> None:
    alg = Algebra.cga3d()
    dh = [(0.0, 1.0, 0.0, 0.0), (0.0, 0.5, 0.0, 0.0), (0.0, 0.3, 0.0, 0.0)]
    target_angles = [0.3, -0.5, 0.7]
    fk_params = [
        (0.0, 1.0, 0.0, target_angles[0]),
        (0.0, 0.5, 0.0, target_angles[1]),
        (0.0, 0.3, 0.0, target_angles[2]),
    ]
    target = robo.fk(alg, fk_params)[-1]["motor"]
    result = robo.ik_dls(
        alg, dh, target,
        position_tolerance=1e-8, orientation_tolerance=1e-8,
    )
    assert result.success
    assert result.position_error < 1e-6
    assert result.orientation_error < 1e-6


def test_ik_dls_identity_target() -> None:
    alg = Algebra.cga3d()
    dh = [(0.0, 1.0, 0.0, 0.0), (0.0, 0.5, 0.0, 0.0)]
    fk_params = [(0.0, 1.0, 0.0, math.pi), (0.0, 0.5, 0.0, 0.0)]
    target = robo.fk(alg, fk_params)[-1]["motor"]
    result = robo.ik_dls(
        alg, dh, target,
        position_tolerance=1e-8, orientation_tolerance=1e-8,
    )
    assert result.success
    assert result.position_error < 1e-6


def test_ik_dls_from_near_singularity() -> None:
    """Solver should escape a near-singular starting configuration."""
    alg = Algebra.cga3d()
    dh = [(0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)]
    fk_params = [(0.0, 1.0, 0.0, 0.5), (0.0, 1.0, 0.0, -0.3)]
    target = robo.fk(alg, fk_params)[-1]["motor"]
    result = robo.ik_dls(
        alg, dh, target,
        initial_angles=np.array([0.01, 0.01]),
        position_tolerance=1e-8, orientation_tolerance=1e-8,
    )
    assert result.success


def test_ik_dls_joint_limits() -> None:
    alg = Algebra.cga3d()
    dh = [(0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)]
    fk_params = [(0.0, 1.0, 0.0, 2.0), (0.0, 1.0, 0.0, -0.5)]
    target_motor = robo.fk(alg, fk_params)[-1]["motor"]
    limits = [(-1.0, 1.0), (-1.0, 1.0)]
    result = robo.ik_dls(
        alg, dh, target_motor,
        joint_limits=limits, position_tolerance=1e-6,
    )
    for i, (lo, hi) in enumerate(limits):
        assert lo - 1e-10 <= result.joint_angles[i] <= hi + 1e-10


def test_ik_dls_unreachable_target() -> None:
    """Target outside the reachable workspace should not converge."""
    alg = Algebra.cga3d()
    dh = [(0.0, 0.5, 0.0, 0.0), (0.0, 0.5, 0.0, 0.0)]
    far_motor = alg.translate([3.0, 0.0, 0.0])
    result = robo.ik_dls(
        alg, dh, far_motor,
        initial_angles=np.array([0.1, 0.1]),
        position_tolerance=1e-6, max_iterations=200,
    )
    assert not result.success or result.position_error > 1e-2


def test_ik_dls_prismatic() -> None:
    alg = Algebra.cga3d()
    dh = [(0.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.0)]
    joint_types = ["prismatic", "revolute"]
    fk_params = [(0.0, 0.0, 0.5, 0.0), (0.0, 1.0, 0.0, 0.3)]
    target = robo.fk(alg, fk_params, joint_types=joint_types)[-1]["motor"]
    result = robo.ik_dls(
        alg, dh, target,
        joint_types=joint_types,
        position_tolerance=1e-6, orientation_tolerance=1e-6,
    )
    assert result.success
    assert_allclose(result.joint_angles[0], 0.5, atol=1e-4)
    assert_allclose(result.joint_angles[1], 0.3, atol=1e-4)


def test_ik_dls_motor_roundtrip() -> None:
    alg = Algebra.cga3d()
    dh = [(0.0, 1.0, 0.0, 0.0), (0.0, 0.6, 0.0, 0.0), (0.0, 0.3, 0.0, 0.0)]
    target_angles = [0.5, -0.3, 0.8]
    fk_params = [
        (0.0, 1.0, 0.0, target_angles[0]),
        (0.0, 0.6, 0.0, target_angles[1]),
        (0.0, 0.3, 0.0, target_angles[2]),
    ]
    target = robo.fk(alg, fk_params)[-1]["motor"]
    result = robo.ik_dls(
        alg, dh, target,
        position_tolerance=1e-8, orientation_tolerance=1e-8,
    )
    assert result.success
    q0, q1, q2 = result.joint_angles[0], result.joint_angles[1], result.joint_angles[2]
    fk_check = robo.fk(
        alg,
        [(0.0, 1.0, 0.0, q0), (0.0, 0.6, 0.0, q1), (0.0, 0.3, 0.0, q2)],
    )
    assert_allclose(fk_check[-1]["position"], result.position, atol=1e-6)


def test_ik_dls_initial_guess_used() -> None:
    alg = Algebra.cga3d()
    dh = [(0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)]
    fk_params = [(0.0, 1.0, 0.0, 0.5), (0.0, 1.0, 0.0, -0.3)]
    target = robo.fk(alg, fk_params)[-1]["motor"]
    result_from_zero = robo.ik_dls(
        alg, dh, target,
        position_tolerance=1e-8, orientation_tolerance=1e-8,
    )
    result_from_near = robo.ik_dls(
        alg, dh, target,
        initial_angles=np.array([0.5, -0.3]),
        position_tolerance=1e-8, orientation_tolerance=1e-8,
    )
    assert result_from_near.success
    assert result_from_near.iterations <= result_from_zero.iterations + 2


def test_ik_dls_nonzero_alpha_joint() -> None:
    alg = Algebra.cga3d()
    dh = [
        (0.0, 0.0, 1.0, 0.0),
        (math.pi / 2, 0.5, 0.0, 0.0),
        (0.0, 0.3, 0.0, 0.0),
    ]
    fk_params = [
        (0.0, 0.0, 1.0, 0.4),
        (math.pi / 2, 0.5, 0.0, 0.6),
        (0.0, 0.3, 0.0, -0.3),
    ]
    target = robo.fk(alg, fk_params)[-1]["motor"]
    result = robo.ik_dls(
        alg, dh, target,
        position_tolerance=1e-6, orientation_tolerance=1e-6,
    )
    assert result.success
    assert result.position_error < 1e-5
    assert result.orientation_error < 1e-5


def test_ik_cga_spherical_wrist_ur5_returns_joint_angles() -> None:
    alg = Algebra.cga3d()
    dh = [
        (math.pi / 2, 0.0, 0.089159, 0.0),
        (0.0, -0.42500, 0.0, 0.0),
        (0.0, -0.39225, 0.0, 0.0),
        (math.pi / 2, 0.0, 0.10915, 0.0),
        (-math.pi / 2, 0.0, 0.09465, 0.0),
        (0.0, 0.0, 0.08230, 0.0),
    ]
    target_angles = np.array([0.45, -1.05, 0.85, -0.35, 0.65, -0.40])
    target = robo.fk(
        alg,
        [
            (alpha, a, d, float(theta))
            for (alpha, a, d, _), theta in zip(dh, target_angles, strict=True)
        ],
    )[-1]["motor"]

    result = robo.ik(
        alg,
        dh,
        target,
        solver="cga_spherical_wrist",
        joint_limits=[(-2.0 * math.pi, 2.0 * math.pi)] * 6,
        position_tolerance=1e-8,
        orientation_tolerance=1e-8,
        max_iterations=200,
    )

    assert result.success
    assert result.joint_angles.shape == (6,)
    assert_allclose(result.joint_angles, target_angles, atol=1e-5)
    assert result.position_error < 1e-8
    assert result.orientation_error < 1e-8
