import numpy as np

from .scene import Elipse, ElipseArc, Line, LineSegment, Scene, SceneError


def _minSides(N):
    if N < 3:
        raise SceneError(f"a regular polygon needs N >= 3 sides, got {N}")


def circle(R=1.0, g=0.0, deadTime=1e-9):
    return Scene(g=g, deadTime=deadTime,
                 solidObjects=[Elipse(0.0, 0.0, R, R, 0.0)])


def oval(a=1.0, b=0.5, tilt=0.0, g=0.0, deadTime=1e-9):
    return Scene(g=g, deadTime=deadTime,
                 solidObjects=[Elipse(0.0, 0.0, a, b, tilt)])


def stadium(r=1.0, a=1.0, g=0.0, deadTime=1e-9):
    return Scene(g=g, deadTime=deadTime, solidObjects=[
        LineSegment(-a, -r, a, -r),
        ElipseArc(a, 0.0, r, r, 0.0, -0.5 * np.pi, 0.5 * np.pi),
        LineSegment(a, r, -a, r),
        ElipseArc(-a, 0.0, r, r, 0.0, 0.5 * np.pi, 1.5 * np.pi),
    ])


def lemon(B=0.5, R=1.0, g=0.0, deadTime=1e-9):
    phi = np.arccos(B / R)
    return Scene(g=g, deadTime=deadTime, solidObjects=[
        ElipseArc(-B, 0.0, R, R, 0.0, -phi, phi),
        ElipseArc(B, 0.0, R, R, 0.0, np.pi - phi, np.pi + phi),
    ])


def mushroom(R=1.0, w=0.5, h=1.0, offset=0.0, g=0.0, deadTime=1e-9):
    s0, s1 = offset - 0.5 * w, offset + 0.5 * w
    return Scene(g=g, deadTime=deadTime, solidObjects=[
        ElipseArc(0.0, 0.0, R, R, 0.0, 0.0, np.pi),
        LineSegment(-R, 0.0, s0, 0.0),
        LineSegment(s0, 0.0, s0, -h),
        LineSegment(s0, -h, s1, -h),
        LineSegment(s1, -h, s1, 0.0),
        LineSegment(s1, 0.0, R, 0.0),
    ])


def poliLine(R, N, theta0=0.0, g=0.0, deadTime=1e-9):
    _minSides(N)
    d = R * np.cos(np.pi / N)
    theta = theta0 + 2.0 * np.pi * (np.arange(N) + 0.5) / N
    return Scene(g=g, deadTime=deadTime, solidObjects=[
        Line(np.cos(t), np.sin(t), -d) for t in theta])


def poliLineSeg(R, N, theta0=0.0, g=0.0, deadTime=1e-9):
    _minSides(N)
    phi = theta0 + 2.0 * np.pi * np.arange(N + 1) / N
    x, y = R * np.cos(phi), R * np.sin(phi)
    return Scene(g=g, deadTime=deadTime, solidObjects=[
        LineSegment(x[k], y[k], x[k + 1], y[k + 1]) for k in range(N)])


def flower(R, N, theta0=0.0, X=0.5, g=0.0, deadTime=1e-9):
    _minSides(N)
    L = 2.0 * R * np.sin(np.pi / N)
    rho = L / (2.0 * abs(X))
    beta = np.arcsin(abs(X))
    c = R * np.cos(np.pi / N) - np.sign(X) * rho * np.cos(beta)
    psi = theta0 + 2.0 * np.pi * (np.arange(N) + 0.5) / N
    omega = psi if X > 0 else psi + np.pi
    return Scene(g=g, deadTime=deadTime, solidObjects=[
        ElipseArc(c * np.cos(p), c * np.sin(p), rho, rho, 0.0, w - beta,
                  w + beta) for p, w in zip(psi, omega)])


def PolyCircle(N, R=1.0, r=0.1, d=0.5, theta0=0.0, g=0.0, deadTime=1e-9):
    phi = theta0 + 2.0 * np.pi * np.arange(N) / N
    return Scene(g=g, deadTime=deadTime, solidObjects=[
        Elipse(0.0, 0.0, R, R, 0.0)] + [
        Elipse(d * np.cos(p), d * np.sin(p), r, r, 0.0) for p in phi])
