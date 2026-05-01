import json
import math

import amsa.robo as robo


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
