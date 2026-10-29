"""Run-time benchmark: one scene, timed over Np and Nobj.

Every scene keeps its particles trapped, so each makes all `iter` events, and
each is built so that a particle only ever meets one object -- while every event
still solves for all Nobj of them. The cost of an event then grows with Nobj.
The scenes themselves come from analyticBilliards.commonTables; gravity g = 1.

    Ellipse     Nobj nested concentric ellipses, the k-th with radii 1.5k and k.
                Particles start inside the innermost and only ever hit it.
    ElipseArc   the same, each arc open within 0.3 rad of the top (phi = pi/2).
                The gap starts at height k cos(0.3) ~ 0.955k and no particle
                climbs above 0.9, so nothing escapes -- but every event also runs
                the arc test of every arc.
    LineSeg     Nobj horizontal segments from x = -1e8 to 1e8, the k-th at
                y = -k. Particles bounce on the top one with vx = 1e-5 and drift
                nowhere near an end.
    Line        LineSeg without the ends: the lines y = -k.
    Portal      a floor from x = -1 to 1 whose side walls, 0 <= y <= 1, are Nobj
                stacked portals, two per strip: periodic in x. No particle climbs
                above 0.9, so there is no ceiling to hit. Nobj counts the
                portals, so it must be even; the floor makes it Nobj + 1.

    python benchmarks/scene.py --scene Ellipse --Np 1000 --Nobj 1 10 100 --iter 1000
    python benchmarks/scene.py --scene Ellipse Line Portal --Nobj 10   # several
"""

import numpy as np

import benchCommon as bc
from analyticBilliards import commonTables as T

# Ellipse, ElipseArc
A, B = 1.5, 1.0  # innermost radii; the k-th is k times these
TOP = 0.9        # no particle climbs above TOP * B (TOP * H for Portal)
GAP = 0.3        # half-width of the arcs' opening, radians of the parametric angle
# LineSeg, Line
L = 1e8
VX = 1e-5
# Portal
H = 1.0


def ellipseICs(n, rng):
    """uniform in the innermost ellipse shrunk by half, random direction, and
    slow enough that y0 + v^2 / 2g stays under TOP * B"""
    u, v = np.empty(0), np.empty(0)
    while u.size < n:
        pu, pv = rng.uniform(-1, 1, 2 * n), rng.uniform(-1, 1, 2 * n)
        keep = pu ** 2 + pv ** 2 < 1
        u, v = np.r_[u, pu[keep]], np.r_[v, pv[keep]]
    x, y = 0.5 * A * u[:n], 0.5 * B * v[:n]
    speed = rng.uniform(0.1, 1.0, n) * np.sqrt(2 * bc.G * (TOP * B - y))
    angle = rng.uniform(0, 2 * np.pi, n)
    return x, y, speed * np.cos(angle), speed * np.sin(angle)


def ellipseConfined(x, y):
    return (x / A) ** 2 + (y / B) ** 2 <= 1 + 1e-9


def stackICs(n, rng):
    return (rng.uniform(-1, 1, n), rng.uniform(0.1, 1, n), np.full(n, VX),
            rng.uniform(-1, 1, n))


def stackConfined(x, y):
    return (y >= -1e-9) & (np.abs(x) < L)


def portalScene(nObj):
    if nObj < 2 or nObj % 2:
        raise SystemExit(f"Portal: --Nobj counts portals, two per strip, so it "
                         f"must be even and >= 2, got {nObj}")
    return T.portalBox(nObj, H, g=bc.G)


def portalICs(n, rng):
    """anywhere in the lower half of the box, random direction, slow enough that
    y0 + v^2 / 2g stays under TOP * H"""
    x, y = rng.uniform(-0.9, 0.9, n), rng.uniform(0.1, 0.5, n)
    speed = rng.uniform(0.1, 1.0, n) * np.sqrt(2 * bc.G * (TOP * H - y))
    angle = rng.uniform(0, 2 * np.pi, n)
    return x, y, speed * np.cos(angle), speed * np.sin(angle)


def portalConfined(x, y):
    return (np.abs(x) <= 1 + 1e-9) & (y >= -1e-9) & (y < H)


# name -> (makeScene(nObj), makeICs(n, rng), confined(x, y), floor: the lowest
# height a particle can reach). The names are what scaling.csv records
SCENES = {
    "Ellipse": (lambda n: T.nestedEllipses(n, A, B, g=bc.G),
                ellipseICs, ellipseConfined, -B),
    "ElipseArc": (lambda n: T.nestedEllipseArcs(n, A, B, GAP, g=bc.G),
                  ellipseICs, ellipseConfined, -B),
    "LineSeg": (lambda n: T.stackedLineSegs(n, L, g=bc.G),
                stackICs, stackConfined, 0.0),
    "Line": (lambda n: T.stackedLines(n, g=bc.G),
             stackICs, stackConfined, 0.0),
    "Portal": (portalScene, portalICs, portalConfined, 0.0),
}


if __name__ == "__main__":
    p = bc.parser(__doc__)
    p.add_argument("--scene", nargs="+", required=True, choices=list(SCENES))
    args = p.parse_args()
    for name in args.scene:
        bc.runBench(name, *SCENES[name], args)
