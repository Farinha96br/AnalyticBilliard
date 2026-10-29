"""The scenes the benchmarks run (benchmarks/, benchmarks/GPU_WORTH/), and the
Lorentz-gas cell of paper_examples/klagesDellago.py.

The stacked and nested families take N objects so the cost of an event can be
scaled: a particle only ever meets the innermost or topmost one, but every event
still solves for all N.
"""

import numpy as np

from ..scene import (Ellipse, EllipseArc, Line, LineSegment, Portal, Scene,
                     SceneError)


def rectangle(a=1.0, b=1.0, g=0.0, deadTime=1e-9):
    """the box [-a, a] x [-b, b], four LineSegments"""
    corners = [(-a, -b), (a, -b), (a, b), (-a, b)]
    return Scene(g=g, deadTime=deadTime, solidObjects=[
        LineSegment(*corners[k], *corners[(k + 1) % 4]) for k in range(4)])


def nestedEllipses(N, a=1.5, b=1.0, g=0.0, deadTime=1e-9):
    """N concentric ellipses, the k-th (k = 1..N) with radii k a and k b"""
    return Scene(g=g, deadTime=deadTime, solidObjects=[
        Ellipse(0.0, 0.0, a * k, b * k, 0.0) for k in range(1, N + 1)])


def nestedEllipseArcs(N, a=1.5, b=1.0, gap=0.3, g=0.0, deadTime=1e-9):
    """nestedEllipses, each open at the top: phi within `gap` of pi/2 is left
    out. phi1 under phi0, so each arc wraps through zero, the long way round"""
    phi0, phi1 = np.pi / 2 + gap, np.pi / 2 - gap
    return Scene(g=g, deadTime=deadTime, solidObjects=[
        EllipseArc(0.0, 0.0, a * k, b * k, 0.0, phi0, phi1)
        for k in range(1, N + 1)])


def stackedLineSegs(N, L=1e8, g=0.0, deadTime=1e-9):
    """N horizontal segments from x = -L to L, the k-th (k = 0..N-1) at y = -k"""
    return Scene(g=g, deadTime=deadTime, solidObjects=[
        LineSegment(-L, -k, L, -k) for k in range(N)])


def stackedLines(N, g=0.0, deadTime=1e-9):
    """stackedLineSegs without the ends: the lines y = -k, k = 0..N-1"""
    return Scene(g=g, deadTime=deadTime, solidObjects=[
        Line(0.0, 1.0, float(k)) for k in range(N)])


def portalBox(N, H=1.0, g=0.0, deadTime=1e-9):
    """a floor from x = -1 to 1 whose side walls, 0 <= y <= H, are N portals:
    N / 2 stacked strips, each a pair of one-way portals between the two sides
    at the same height. Periodic in x; there is no ceiling"""
    if N < 2 or N % 2:
        raise SceneError(f"portalBox needs an even N >= 2 (two portals per "
                         f"strip), got {N}")
    edges = np.linspace(0.0, H, N // 2 + 1)
    portals = []
    for y0, y1 in zip(edges[:-1], edges[1:]):
        right = LineSegment(1.0, y0, 1.0, y1)
        left = LineSegment(-1.0, y0, -1.0, y1)
        portals += [Portal(right, left), Portal(left, right)]
    return Scene(g=g, deadTime=deadTime,
                 solidObjects=[LineSegment(-1.0, 0.0, 1.0, 0.0)],
                 portalObjects=portals)


def lorentzGas(w, R=1.0, g=0.0, deadTime=1e-9):
    """the rectangular cell of the triangular Lorentz gas: discs of radius R
    whose gaps are w, a whole one in the middle and a quarter in each corner,
    the cell made periodic by portals on its four sides"""
    a = 2 * R + w
    hx, hy = a / 2, a * np.sqrt(3) / 2
    # each corner keeps only the quarter of its disc that lies inside the cell
    corners = [(hx, hy, np.pi, 1.5 * np.pi), (-hx, hy, 1.5 * np.pi, 2 * np.pi),
               (-hx, -hy, 0.0, 0.5 * np.pi), (hx, -hy, 0.5 * np.pi, np.pi)]
    right = LineSegment(hx, -hy, hx, hy)
    left = LineSegment(-hx, -hy, -hx, hy)
    top = LineSegment(-hx, hy, hx, hy)
    bottom = LineSegment(-hx, -hy, hx, -hy)
    return Scene(
        g=g, deadTime=deadTime,
        solidObjects=[Ellipse(0.0, 0.0, R, R, 0.0)]
        + [EllipseArc(cx, cy, R, R, 0.0, p0, p1) for cx, cy, p0, p1 in corners],
        # same orientation on both sides and no flips: a pure translation
        portalObjects=[Portal(right, left), Portal(left, right),
                       Portal(top, bottom), Portal(bottom, top)])
