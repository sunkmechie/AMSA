import numpy as np
import pytest

from amsa import MVArray, MVLayout, vga


def test_dense_layout_covers_full_basis() -> None:
    spec = vga(2)
    layout = MVLayout.dense(spec)
    assert layout.blades == (0, 1, 2, 3)
    assert layout.kind == "dense"
    assert layout.grades == (0, 1, 2)


def test_grade_layout_is_grade_packed() -> None:
    spec = vga(3)
    layout = MVLayout.grade(spec, 1, 2)
    assert layout.blades == (1, 2, 4, 3, 5, 6)
    assert layout.grades == (1, 2)


def test_sparse_layout_requires_unique_blades() -> None:
    spec = vga(2)
    with pytest.raises(ValueError):
        MVLayout.sparse_pattern(spec, (1, 1))


def test_mvarray_validates_shape_against_layout() -> None:
    spec = vga(2)
    layout = MVLayout.grade(spec, 1)
    values = np.zeros((5, layout.size))
    mv = MVArray(algebra=spec, layout=layout, values=values)
    assert mv.batch_shape == (5,)

    with pytest.raises(ValueError):
        MVArray(algebra=spec, layout=layout, values=np.zeros((layout.size + 1,)))
