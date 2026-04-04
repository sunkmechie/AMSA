import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for amsa.viz.backends.mpl. "
        "Install it via `pip install matplotlib`."
    ) from exc

from amsa.viz.primitives import Point


def plot(ax: Axes, primitive: Point, **kwargs) -> None:
    """
    Plots a viz primitive onto a matplotlib Axes.
    """
    if isinstance(primitive, Point):
        pos = primitive.position
        # Extract base styling from the primitive, override with kwargs if provided
        color = kwargs.pop("color", primitive.color)
        label = kwargs.pop("label", primitive.label)

        # Plot based on dimension
        if pos.shape[-1] == 2:
            if pos.ndim == 1:
                ax.scatter(pos[0], pos[1], color=color, label=label, **kwargs)
            else:
                ax.scatter(pos[..., 0], pos[..., 1], color=color, label=label, **kwargs)
        elif pos.shape[-1] == 3:
            if not hasattr(ax, "zaxis"):
                raise ValueError("Cannot plot 3D point on a 2D matplotlib Axes.")
            if pos.ndim == 1:
                ax.scatter(pos[0], pos[1], pos[2], color=color, label=label, **kwargs)
            else:
                ax.scatter(
                    pos[..., 0], pos[..., 1], pos[..., 2], color=color, label=label, **kwargs
                )
        else:
            raise ValueError(f"Unsupported point dimension: {pos.shape[-1]}")
    else:
        raise NotImplementedError(f"Plotting for {type(primitive)} is not implemented yet in mpl.")


def show() -> None:
    """Convenience wrapper for plt.show()"""
    plt.show()
