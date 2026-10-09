import numpy as np
from matplotlib import colormaps

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


def _segment(params):
    x0, y0, x1, y1 = params
    return np.array([x0, x1]), np.array([y0, y1])


def _ellipse(kind, params):
    cx, cy, a, b, tilt = params[:5]
    phi0, phi1 = params[5:7] if kind == "elipseArc" else (0.0, 2.0 * np.pi)
    if phi1 <= phi0:
        phi1 += 2.0 * np.pi

    phi = np.linspace(phi0, phi1, ARC_POINTS)
    u, v = a * np.cos(phi), b * np.sin(phi)
    return (cx + u * np.cos(tilt) - v * np.sin(tilt),
            cy + u * np.sin(tilt) + v * np.cos(tilt))


def _infiniteLine(params, span):
    a, b, c = params
    footX, footY = -a * c, -b * c
    return (np.array([footX - b * span, footX + b * span]),
            np.array([footY + a * span, footY - a * span]))


def _curve(kind, params):
    return _segment(params) if kind == "lineSegment" else _ellipse(kind, params)


def _parts(scene):
    parts = [(o["type"], o["params"], WALL_STYLE)
             for o in scene.get("solidObjects", [])]

    for k, portal in enumerate(scene.get("portalObjects", [])):
        parts.append(("lineSegment", portal["entryParams"],
                      {**PORTAL_STYLE, "label": f"portal {k + 1}",
                       "color": PORTAL_COLORS[k % len(PORTAL_COLORS)]}))

    basins = scene.get("basinObjects") or {}
    for k, label in enumerate(sorted(basins)):
        color = BASIN_COLORS[k % len(BASIN_COLORS)]
        for j, o in enumerate(basins[label]):
            parts.append((o["type"], o["params"],
                          {**BASIN_STYLE, "color": color,
                           "label": f"basin {label}" if j == 0 else None}))
    return parts


def _drawInfiniteLines(axes, lines):
    (xLo, xHi), (yLo, yHi) = axes.get_xlim(), axes.get_ylim()
    for (a, b, c), _ in lines:
        xLo, xHi = min(xLo, -a * c), max(xHi, -a * c)
        yLo, yHi = min(yLo, -b * c), max(yHi, -b * c)

    span = 2.0 * np.hypot(xHi - xLo, yHi - yLo)
    for params, style in lines:
        axes.plot(*_infiniteLine(params, span), **style)

    axes.set_xlim(xLo, xHi)
    axes.set_ylim(yLo, yHi)


def plotScene(scene, axes):
    parts = _parts(scene)

    for kind, params, style in parts:
        if kind != "line":
            axes.plot(*_curve(kind, params), solid_capstyle="butt", **style)

    lines = [(params, style) for kind, params, style in parts if kind == "line"]
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
