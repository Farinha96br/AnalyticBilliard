import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hpcBilliards import compileScene, updateScene



R = 1.0
G = 0.5
H = -1.01   # the line's HEIGHT. a*x + b*y + c = 0 with (0,1,c) puts it at
            # y = -c, so the c passed below is -H and H reads as a height

# there is no "circle" object type: a circle is an ellipse with both radii the
# same, and no rotation. params are (x0, y0, a, b, theta)
SCENE = {"g": G,  # gravity
         "deadTime": 1e-9,
            "solidObjects": [
                {"type": "elipse", "params": [0.0, 0.0, R, R, 0.0]}, # circle with radius R
                {"type": "line", "params": [0.0, 1.0, H]}
            ]
        }



def startWithFixedEnergy(E,N,R,g=1.0,seed=0,floor=-np.inf):

    
    rng = np.random.default_rng(seed)

    # The band of heights a particle can both OCCUPY (inside the circle, above
    # the line) and REACH (0.5*v^2 = E - g*y must not be negative).

    lowest = max(floor, -R)
    highest = min(E / g, R) if g != 0.0 else R

    if not highest > lowest:
        raise ValueError(
            f"nothing to sample: the line at y={floor} is at or above the "
            f"highest point E={E} can reach (y={E / g if g else np.inf:.4f}). "
            f"no state has this energy inside this billiard")

    x = np.empty(N)
    y = np.empty(N)
    accepted = 0
    attempts = 0

    while accepted < N:
        batch = max(4 * (N - accepted), 1024)
        attempts += 1
        if attempts > 1000:
            raise RuntimeError(f"gave up sampling after {attempts} batches: the "
                               f"band y in [{lowest:.4f}, {highest:.4f}] is too "
                               f"thin a sliver of the circle to fill")

        # uniform by AREA over the band: draw the height uniformly, then keep it
        # with probability proportional to the circle's width there. without the
        # weighting the ends of the band would be over-represented
        candidateY = rng.uniform(lowest, highest, batch)
        halfChord = np.sqrt(np.maximum(R * R - candidateY * candidateY, 0.0))
        candidateY = candidateY[rng.random(batch) * R <= halfChord]

        halfChord = np.sqrt(np.maximum(R * R - candidateY * candidateY, 0.0))
        candidateX = (2.0 * rng.random(candidateY.size) - 1.0) * halfChord

        room = min(N - accepted, candidateX.size)
        x[accepted:accepted + room] = candidateX[:room]
        y[accepted:accepted + room] = candidateY[:room]
        accepted += room

    speed = np.sqrt(2.0 * (E - g * y))
    direction = rng.uniform(0.0, 2.0 * np.pi, N)

    vx = speed * np.cos(direction)
    vy = speed * np.sin(direction)

    return x, y, vx, vy



def seriesToMap(x,y,vx,vy,lineY=None):
    """
    collisions -> (theta, alpha).

    alpha is measured from the TANGENT of whatever surface was hit, so when a
    line is present the collisions on it need its own tangent: horizontal, not
    perpendicular to the radius. Using the circle's tangent for a hit on the
    chord measures alpha against a surface that is not there.
    """
    theta = np.arctan2(y, x) % (2.0 * np.pi)

    # the circle's counterclockwise tangent everywhere...
    tangentX, tangentY = -np.sin(theta), np.cos(theta)

    # ...except on the chord, where the surface is flat. counterclockwise means
    # interior on the left, and the interior is above, so the tangent is +x
    if lineY is not None:
        onLine = np.abs(y - lineY) < 1e-7
        tangentX = np.where(onLine, 1.0, tangentX)
        tangentY = np.where(onLine, 0.0, tangentY)

    # the inward normal is the tangent turned a quarter turn, on either surface
    normalX, normalY = -tangentY, tangentX

    alongTangent = vx * tangentX + vy * tangentY
    alongNormal = vx * normalX + vy * normalY

    alpha = np.arctan2(alongNormal, alongTangent) % (2.0 * np.pi)

    return theta, alpha


def lineThetaSpan(lineY, R=R):
    """
    the theta interval whose collisions land on the chord rather than the arc.

    The chord runs between the two corners, at theta = cut and pi - cut, and
    which way round it goes is decided by its own midpoint (0, lineY): that
    sits at theta = pi/2 when the line is above the centre and 3*pi/2 when it
    is below. NOT the complement of "the circle is above the line" -- points on
    the chord are interior points, so the circle's own relation does not apply
    to them.
    """
    cut = np.arcsin(np.clip(lineY / R, -1.0, 1.0))
    if lineY >= 0.0:
        return [(cut, np.pi - cut)]                    # over the top
    return [(np.pi - cut, 2.0 * np.pi + cut)]          # under the bottom



#E_paper = 2, 1.416, 1.088, 0.72, 0.3, 0.1 at g = 0.5.
def paperEnergyToHere(E_paper, R=R, g=G):
    return E_paper - g * R


def hereEnergyToPaper(E_here, R=R, g=G):
    return E_here + g * R



def drawBilliard(axes, lineY):
    """the arc above the line, and the chord itself"""
    cut = np.arcsin(np.clip(lineY / R, -1.0, 1.0))

    # the circle is drawn whole when the line misses it, otherwise only the
    # part that is really a wall
    if abs(lineY) < R:
        arc = np.linspace(cut, np.pi - cut, 400)
        halfChord = np.sqrt(R * R - lineY * lineY)
        axes.plot([-halfChord, halfChord], [lineY, lineY],
                  color="tab:red", lw=2, zorder=2)
    else:
        arc = np.linspace(0.0, 2.0 * np.pi, 400)
    axes.plot(R * np.cos(arc), R * np.sin(arc), color="0.3", lw=2, zorder=2)


def drawTrajectory(axes, run, particle, g, nArcs=60):
    """
    the flight path between collisions, rebuilt from the closed form.

    Every row is the state a flight STARTS from and the next row is where it
    lands, so the arc between them is just the parabola sampled for drawing.
    """
    rows = min(int(run.counts[particle]), nArcs + 1)
    t = run.t[particle, :rows]
    x, y = run.x[particle, :rows], run.y[particle, :rows]
    vx, vy = run.vx[particle, :rows], run.vy[particle, :rows]

    for i in range(rows - 1):
        step = np.linspace(0.0, t[i + 1] - t[i], 40)
        axes.plot(x[i] + vx[i] * step,
                  y[i] + vy[i] * step - 0.5 * g * step * step,
                  color=plt.cm.plasma(0.85 * i / max(rows - 2, 1)),
                  lw=0.9, alpha=0.9, zorder=3)

    axes.plot(x[0], y[0], "*", ms=13, color=plt.cm.plasma(0.0),
              mec="black", mew=0.7, zorder=4)


NPARTICLES = 80
NBOUNCES = 10000

TRAJECTORY_ENERGIES = [5.0, 1.088, 0.72]   # one panel each in the path figure
TRAJECTORY_ARCS = 60

# the six energies of Fig. 2, in the paper's own convention
paperEnergies = [5.0, 1.416, 1.088, 0.72, 0.3, 0.1]

OUTDIR = pathlib.Path("mapCircle")
OUTDIR.mkdir(exist_ok=True)

billiard = compileScene(SCENE, backend="openmp")


# lineY IS the height of the line, and the only place the sign convention gets
# applied is the "params" below: a*x + b*y + c = 0 with (0, 1, c) sits at
# y = -c, so c is -lineY. Keeping the height in its own variable is what stops
# the direction flipping around by accident.
#
# the line starts just under the circle and rises through it
lineHeights = np.linspace(-1.01, -0.9, 15)

for frame, lineY in enumerate(lineHeights):

    # the scene changes with lineY and not with E, so it is set once per figure.
    # updateScene takes (scene, compiled) in that order, and the dict needs g
    # and deadTime: anything left out falls back to the library default, so a
    # missing "g" would quietly run this at g = 1.0 instead of 0.5
    updateScene({"g": G, "deadTime": 1e-9,
                 "solidObjects": [
                     {"type": "elipse", "params": [0.0, 0.0, R, R, 0.0]},
                     {"type": "line", "params": [0.0, 1.0, -lineY]}
                 ]}, billiard)

    # the stretch of theta that belongs to the chord rather than the arc, and
    # the two angles where they meet
    lineSpans = lineThetaSpan(lineY, R) if abs(lineY) < R else []
    cut = np.arcsin(np.clip(lineY / R, -1.0, 1.0))
    cutLeft, cutRight = (np.pi - cut) % (2 * np.pi), cut % (2 * np.pi)

    figure, axesGrid = plt.subplots(2, 3, figsize=(14, 8), facecolor="white")
    for axes, E_paper in zip(axesGrid.ravel(), paperEnergies):
        E_here = paperEnergyToHere(E_paper)

        axes.set_xlim(0.0, 2.0 * np.pi)
        axes.set_ylim(0.0, np.pi)
        axes.set_xlabel(r"$\theta$")
        axes.set_ylabel(r"$\alpha$")
        axes.set_title(f"E = {E_paper}", loc="left", fontsize=11)

        # raising the line eventually lifts the floor above everything this
        # energy can reach, and then the billiard simply has no states left.
        # that is physics, not a failure, so say so and move to the next panel
        try:
            x0, y0, vx0, vy0 = startWithFixedEnergy(E=E_here, N=NPARTICLES, R=R,
                                                    g=G, seed=0, floor=lineY)
        except ValueError:
            axes.text(0.5, 0.5, "no states at this energy\nabove the line",
                      transform=axes.transAxes, ha="center", va="center",
                      fontsize=11, color="0.5")
            continue

        # no "ID" column: the particle index is the first axis of every array, so
        # run.x has shape (nParticles, rows) and particle i is run.x[i]
        run = billiard.recordWithIterations(x0, y0, vx0, vy0, iterations=NBOUNCES,
                                save=["x", "y", "vx", "vy"],
                                eventType=["bounce"])

        # nothing may end up under the line. a collision below it would mean a
        # particle leaked through a corner, and its whole orbit after that is
        # bouncing around in a region that is not part of the billiard
        below = run.y.ravel() < lineY - 1e-9
        if below.any():
            print(f"  WARNING lineY={lineY:+.3f} E={E_paper}: "
                  f"{int(below.sum())} collisions below the line")

        # every particle used the same number of rows here, so the whole 2-D block
        # can go through at once -- ravel just stacks all the trajectories together
        theta, alpha = seriesToMap(run.x.ravel(), run.y.ravel(),
                                run.vx.ravel(), run.vy.ravel(), lineY=lineY)

        # which surface each collision landed on. theta cannot be used to tell
        # them apart: with the line ABOVE the centre the surviving arc is the
        # top cap, which spans exactly the same theta as the chord does. So the
        # line part is marked by colour, and the corner angles are drawn on top
        onLine = np.abs(run.y.ravel() - lineY) < 1e-7

        axes.plot(theta[~onLine], alpha[~onLine], ",", color="black", alpha=0.5)
        axes.plot(theta[onLine], alpha[onLine], ",", color="tab:red", alpha=0.5)

        for corner in ([cutLeft, cutRight] if abs(lineY) < R else []):
            axes.axvline(corner, color="tab:blue", lw=0.9, ls="--", alpha=0.8)

    corners = (f"dashed: the corner angles $\\theta$ = {cutRight:.3f} and "
               f"{cutLeft:.3f}" if abs(lineY) < R else "the line misses the circle")
    figure.suptitle(f"circular billiard in gravity with a line at y = {lineY:+.3f}, "
                    f"g = {G}, R = {R}   "
                    f"({NPARTICLES} particles x {NBOUNCES} collisions)\n"
                    f"red: collisions on the line   black: on the arc   |   {corners}",
                    fontsize=11)
    figure.tight_layout()
    # indexed and zero-padded so the frames sort in sweep order
    name = OUTDIR / f"map_{frame:02d}_y{lineY:+.3f}.png"
    figure.savefig(name, dpi=150, facecolor="white")
    plt.close(figure)

    # ---- and the same frame in real space, so the map has something to be
    # read against: what the billiard actually looks like at this line height
    pathFigure, pathAxes = plt.subplots(1, len(TRAJECTORY_ENERGIES),
                                        figsize=(13, 4.6), facecolor="white")

    for axes, E_paper in zip(np.atleast_1d(pathAxes), TRAJECTORY_ENERGIES):
        axes.set_aspect("equal")
        axes.set_xlim(-1.15, 1.15)
        axes.set_ylim(-1.15, 1.15)
        axes.set_facecolor("white")
        axes.set_title(f"E = {E_paper}", loc="left", fontsize=11)
        axes.tick_params(colors="0.4", length=0)
        for side in ("top", "right", "left", "bottom"):
            axes.spines[side].set_visible(False)

        drawBilliard(axes, lineY)

        try:
            x0, y0, vx0, vy0 = startWithFixedEnergy(E=paperEnergyToHere(E_paper),
                                                    N=1, R=R, g=G, seed=3,
                                                    floor=lineY)
        except ValueError:
            axes.text(0.0, 0.0, "no states\nat this energy", ha="center",
                      va="center", fontsize=10, color="0.5")
            continue

        # t is needed here and not in the map: an arc is rebuilt by integrating
        # from one row for exactly t[i+1] - t[i]
        path = billiard.recordWithIterations(x0, y0, vx0, vy0,
                                             iterations=TRAJECTORY_ARCS,
                                             save=["t", "x", "y", "vx", "vy"])
        drawTrajectory(axes, path, 0, G, TRAJECTORY_ARCS)

    pathFigure.suptitle(f"trajectories at y = {lineY:+.3f}   "
                        f"(first {TRAJECTORY_ARCS} flights, dark blue to orange; "
                        f"star = start)", fontsize=11)
    pathFigure.tight_layout()
    pathName = OUTDIR / f"paths_{frame:02d}_y{lineY:+.3f}.png"
    pathFigure.savefig(pathName, dpi=150, facecolor="white")
    plt.close(pathFigure)

    print(f"wrote {name.name} and {pathName.name}")
