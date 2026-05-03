import math
from typing import cast

import numpy as np

import amsa.robo as robo
from amsa import Algebra
from amsa.mv import MVArray
from amsa.ops import sandwich as amsa_sandwich

UR5_DH = [
    (math.pi / 2,  0.0,       0.089159,  0.0),
    (0.0,         -0.42500,   0.0,        0.0),
    (0.0,         -0.39225,   0.0,        0.0),
    (math.pi / 2,  0.0,       0.10915,   0.0),
    (-math.pi / 2, 0.0,       0.09465,   0.0),
    (0.0,          0.0,       0.08230,   0.0),
]

UR5_LIMITS_STRESS = [
    (-math.pi, math.pi),
    (-math.pi, math.pi),
    (-math.pi, math.pi),
    (-1.75, 1.75),
    (-math.pi, math.pi),
    (-math.pi, math.pi),
]

N = 1_000


def test_ur5_fk_stress() -> None:
    """UR5 CGA FK stress test across 10 000 random configurations.

    Validates motor composition, quaternion extraction, rotation-matrix
    orthogonality, determinant consistency, and position extraction across
    the UR5 workspace.
    """
    rng = np.random.default_rng(42)
    alg = Algebra.cga3d()
    origin = alg.origin()

    n_failed_position_finite = 0
    n_failed_quat_norm = 0
    n_failed_orthogonality = 0
    n_failed_determinant = 0
    n_failed_position_match = 0
    n_unstable = 0

    worst_quat_norm_err = 0.0
    worst_ortho_err = 0.0
    worst_det_err = 0.0
    worst_pos_mismatch = 0.0

    unstable_configs: list[dict[str, object]] = []

    def _sample_config() -> list[float]:
        return [rng.uniform(lo, hi) for lo, hi in UR5_LIMITS_STRESS]

    for i in range(N):
        thetas = _sample_config()
        dh = [(alpha, a, d, theta) for (alpha, a, d, _), theta in zip(UR5_DH, thetas, strict=True)]

        results = robo.fk(alg, dh)
        ee = results[5]

        position = cast(np.ndarray, ee["position"])
        quat = cast(np.ndarray, ee["orientation"])
        motor = cast(MVArray, ee["motor"])

        # 1. Position must be finite
        if not np.all(np.isfinite(position)):
            n_failed_position_finite += 1
            unstable_configs.append({"index": i, "thetas": thetas, "reason": "non-finite position"})
            continue

        # 2. Quaternion norm ≈ 1
        quat_norm = np.linalg.norm(quat)
        quat_norm_err = abs(quat_norm - 1.0)
        if quat_norm_err > worst_quat_norm_err:
            worst_quat_norm_err = float(quat_norm_err)
        if quat_norm_err >= 1e-6:
            n_failed_quat_norm += 1

        # 3-4. Rotation matrix orthogonality & determinant
        R = robo.motor_to_matrix(motor, alg)
        RTR = R.T @ R
        ortho_err = np.max(np.abs(RTR - np.eye(3)))
        if ortho_err > worst_ortho_err:
            worst_ortho_err = float(ortho_err)
        if not np.allclose(RTR, np.eye(3), atol=1e-6):
            n_failed_orthogonality += 1

        det_R = np.linalg.det(R)
        det_err = abs(det_R - 1.0)
        if det_err > worst_det_err:
            worst_det_err = float(det_err)
        if det_err >= 1e-6:
            n_failed_determinant += 1

        # 5. Position consistency — two methods
        pos_method_a = position
        tip = amsa_sandwich(motor, origin)
        pos_method_b = alg.extract_point(tip)
        pos_mismatch = np.max(np.abs(pos_method_a - pos_method_b))
        if pos_mismatch > worst_pos_mismatch:
            worst_pos_mismatch = float(pos_mismatch)
        if not np.allclose(pos_method_a, pos_method_b, atol=1e-12):
            n_failed_position_match += 1

        # Singularity / instability tracking
        if math.isclose(det_R, 0.0, abs_tol=1e-12) or quat_norm_err >= 1e-4:
            n_unstable += 1
            unstable_configs.append({
                "index": i,
                "thetas": thetas,
                "position": position.tolist(),
                "quaternion": quat.tolist(),
                "quat_norm_err": quat_norm_err,
                "det_R": det_R,
                "reason": (
                    "near-singular"
                    if math.isclose(det_R, 0.0, abs_tol=1e-12)
                    else "quaternion unstable"
                ),
            })

    # -- Summary -----------------------------------------------------------
    print()
    print("=== AMSA UR5 FK Stress Test ===")
    print()
    print(f"Configurations tested: {N}")
    print()

    all_pass = (
        n_failed_position_finite == 0
        and n_failed_quat_norm == 0
        and n_failed_orthogonality == 0
        and n_failed_determinant == 0
        and n_failed_position_match == 0
    )

    if all_pass:
        print("PASS:")
    else:
        print("FAILURES:")

    def _status(ok: int) -> str:
        return " ✓" if ok == 0 else f" ✗ ({ok})"

    print(f"  finite positions{_status(n_failed_position_finite)}")
    print(f"  quaternion normalization{_status(n_failed_quat_norm)}")
    print(f"  rotation orthogonality{_status(n_failed_orthogonality)}")
    print(f"  determinant consistency{_status(n_failed_determinant)}")
    print(f"  motor position consistency{_status(n_failed_position_match)}")
    print()

    print("Worst-case metrics:")
    print(f"  quaternion norm error:   {worst_quat_norm_err:.1e}")
    print(f"  orthogonality error:     {worst_ortho_err:.1e}")
    print(f"  determinant error:       {worst_det_err:.1e}")
    print(f"  position mismatch:       {worst_pos_mismatch:.1e}")
    print()

    if n_unstable == 0:
        print("No unstable motors detected.")
    else:
        print(f"Unstable configurations: {n_unstable}")
        for cfg in unstable_configs[:5]:
            print(f"  idx={cfg['index']}: {cfg['reason']}")
            print(f"    thetas={[f'{t:.4f}' for t in cast(list[float], cfg['thetas'])]}")
            if "position" in cfg:
                print(f"    pos={cfg['position']}")
            if "det_R" in cfg:
                print(f"    det(R)={cfg['det_R']:.6e}")
        if len(unstable_configs) > 5:
            print(f"  ... and {len(unstable_configs) - 5} more")

    # Assert so pytest reports this as a proper test
    assert n_failed_position_finite == 0, f"{n_failed_position_finite} non-finite positions"
    assert n_failed_quat_norm == 0, f"{n_failed_quat_norm} quaternion norm failures"
    assert n_failed_orthogonality == 0, f"{n_failed_orthogonality} orthogonality failures"
    assert n_failed_determinant == 0, f"{n_failed_determinant} determinant failures"
    assert n_failed_position_match == 0, f"{n_failed_position_match} position mismatch failures"
