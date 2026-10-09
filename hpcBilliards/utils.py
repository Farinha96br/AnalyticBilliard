import numpy as np
from matplotlib import colormaps

from .scene import Elipse, ElipseArc, Line, LineSegment, SceneError

TRAJECTORY_CMAP = "rainbow"
DARKEN = 0.8

WALL_COLOR = "#101010"
PORTAL_COLORS = ["#e8730a", "#1f77b4", "#7a3fa0", "#8a6d3b"]
BASIN_COLORS = ["#c1272d", "#1baf7a", "#00420d", "#b8860b"]

WALL_STYLE = {"color": WALL_COLOR, "lw": 1.5, "zorder": 3}
PORTAL_STYLE = {"lw": 3.0, "zorder": 5}
BASIN_STYLE = {"lw": 3.0, "zorder": 4}

ARC_POINTS = 361

START_MARKER, END_MARKER = "^", "X"
MARKER_STYLE = {"ls": "none", "ms": 6, "mec": "#101010", "mew": 0.1,
                "zorder": 6}


def _segment(o):
    return np.array([o.x0, o.x1]), np.array([o.y0, o.y1])


def _ellipse(o):
    """a whole ellipse, or an arc of one, as a polyline"""
    phi0, phi1 = ((o.phi0, o.phi1) if isinstance(o, ElipseArc)
                  else (0.0, 2.0 * np.pi))
    if phi1 <= phi0:
        phi1 += 2.0 * np.pi  # through zero, not backwards

    phi = np.linspace(phi0, phi1, ARC_POINTS)
    u, v = o.a * np.cos(phi), o.b * np.sin(phi)
    return (o.cx + u * np.cos(o.tilt) - v * np.sin(o.tilt),
            o.cy + u * np.sin(o.tilt) + v * np.cos(o.tilt))


def _infiniteLine(o, span):
    footX, footY = -o.a * o.c, -o.b * o.c
    return (np.array([footX - o.b * span, footX + o.b * span]),
            np.array([footY + o.a * span, footY - o.a * span]))


# exact types, no default: an unknown shape must raise, not fall through
CURVE = {LineSegment: _segment, Elipse: _ellipse, ElipseArc: _ellipse}


def _curve(o):
    try:
        return CURVE[type(o)](o)
    except KeyError:
        raise SceneError(f"plotScene cannot draw {type(o).__name__}") from None


def _parts(scene):
    """-> [(shape, style)]. Line is left for the caller"""
    parts = [(o, WALL_STYLE) for o in scene.solidObjects]

    for k, portal in enumerate(scene.portalObjects):
        parts.append((portal.entry,
                      {**PORTAL_STYLE, "label": f"portal {k + 1}",
                       "color": PORTAL_COLORS[k % len(PORTAL_COLORS)]}))

    for k, label in enumerate(sorted(scene.basinObjects)):
        color = BASIN_COLORS[k % len(BASIN_COLORS)]
        for j, o in enumerate(scene.basinObjects[label]):
            parts.append((o, {**BASIN_STYLE, "color": color,
                              "label": f"basin {label}" if j == 0 else None}))
    return parts


def _drawInfiniteLines(axes, lines):
    (xLo, xHi), (yLo, yHi) = axes.get_xlim(), axes.get_ylim()
    for o, _ in lines:
        xLo, xHi = min(xLo, -o.a * o.c), max(xHi, -o.a * o.c)
        yLo, yHi = min(yLo, -o.b * o.c), max(yHi, -o.b * o.c)

    span = 2.0 * np.hypot(xHi - xLo, yHi - yLo)
    for o, style in lines:
        axes.plot(*_infiniteLine(o, span), **style)

    axes.set_xlim(xLo, xHi)
    axes.set_ylim(yLo, yHi)


def plotScene(scene, axes):
    parts = _parts(scene)

    for o, style in parts:
        if not isinstance(o, Line):
            axes.plot(*_curve(o), solid_capstyle="butt", **style)

    lines = [(o, style) for o, style in parts if isinstance(o, Line)]
    if lines:
        _drawInfiniteLines(axes, lines)

    axes.set_aspect("equal")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    if axes.get_legend_handles_labels()[0]:
        axes.legend(loc="best", fontsize=9, framealpha=0.85)
    return axes


def _rainbow(n):
    shades = colormaps[TRAJECTORY_CMAP](np.linspace(0.0, 1.0, n))
    return DARKEN * shades[:, :3]


def _parabolas(x, y, vx, vy, resolution, g):
    flight = (x[1:] - x[:-1]) / vx[:-1]
    tau = np.linspace(0.0, 1.0, resolution)[None, :] * flight[:, None]

    px = x[:-1, None] + vx[:-1, None] * tau
    py = y[:-1, None] + vy[:-1, None] * tau - 0.5 * g * tau * tau

    gap = np.full((px.shape[0], 1), np.nan)
    return np.hstack([px, gap]).ravel(), np.hstack([py, gap]).ravel()


def plotTrajectory(axes, x, y, vx, vy, resolution, g, particles, interval,
                   **style):
    x, y, vx, vy = (np.asarray(a, dtype=float) for a in (x, y, vx, vy))
    rows = slice(*interval)
    colors = _rainbow(len(particles))

    for i, color in zip(particles, colors):
        px, py = x[i, rows], y[i, rows]

        axes.plot(*_parabolas(px, py, vx[i, rows], vy[i, rows], resolution, g),
                  **{"color": color, **style})
        axes.plot(px[0], py[0], START_MARKER, color=color, **MARKER_STYLE)
        axes.plot(px[-1], py[-1], END_MARKER, color=color, **MARKER_STYLE)
    return axes


def toBirkoff(x, y, vx, vy, x0=0.0, y0=0.0):
    u, v = np.asarray(x) - x0, np.asarray(y) - y0

    wall = np.argmin([v, 1.0 - u, 1.0 - v, u], axis=0)
    theta = wall + np.choose(wall, [u, v, 1.0 - u, 1.0 - v])
    alpha = np.arctan2(vy, vx) - wall * (0.5 * np.pi)

    return theta, np.cos(alpha)
