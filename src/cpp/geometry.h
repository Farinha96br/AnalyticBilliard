#pragma once

#include "types.h"
#include <cmath>


enum ObjectType { LINE = 0,
                  LINE_SEGMENT = 1,
                  ELLIPSE = 2,
                  ELLIPSE_ARC = 3};

enum Response { REFLECT = 0,
                PORTAL = 1,
                ABSORB = 2};

struct Object {
    int type;
    int response;
    double p[13]; // LINE:         a, b, c of a x + b y + c = 0, (a,b) kept unit
                  // LINE_SEGMENT: x0, y0, x1, y1
                  // ELLIPSE:       x0, y0, a, b, theta, cos(theta), sin(theta)
                  // ELLIPSE_ARC:   the seven above, then
                  //               phi0, phi1, cos(phi0), sin(phi0), cos(phi1), sin(phi1)
                  //
                  // an arc's first seven slots are byte for byte an ELLIPSE, which is
                  // why every ellipse* function below reads an arc without conversion

    double q[10]; // PORTAL: a0x, a0y, a1x, a1y  this portal's own frame
                  //         b0x, b0y, b1x, b1y  the frame it leads to
                  //         flipNormal, flipTangent
                  //
                  // the partner is held as coordinates, never as an index: nothing
                  // to dereference on the device
                  //
                  // ABSORB: the basin label, and nothing else. several shapes may
                  //         carry the same one, which is what makes a basin able
                  //         to be a square rather than only a line
};

#ifdef _OPENMP
#pragma omp begin declare target
#endif

// ------------------------------------------------------------- 0. ballistics

inline state trajectory(state s, double t) {
    return {s.x + s.vx * t,
            s.y + s.vy * t - 0.5 * g * t * t,
            s.vx,
            s.vy - g * t};
}

// ---------------------------------------------------------- 1. root finding

inline double firstRoot(double t1, double t2) {
    double t = MAGIC_NO_COLLISION;
    if (t1 > deadTime && t1 < t) t = t1;
    if (t2 > deadTime && t2 < t) t = t2;
    return t;
}

inline vector2D quadRoots(double A, double B, double C) {

    // named, not written inline at each return: nvc++ 26.3 miscompiles an early
    // "return {CONST, CONST};" in device code and hands back zeros, which read as
    // a collision at t = 0 instead of no collision at all
    vector2D none = {MAGIC_NO_COLLISION, MAGIC_NO_COLLISION};

    if (A == 0.0) {
        if (B == 0.0) return none;
        return {-C / B, MAGIC_NO_COLLISION};
    }

    double disc = B * B - 4.0 * A * C;
    if (disc < 0.0) return none;

    // q takes the sign of B so that B + sq never cancels
    double sq = sqrt(disc);
    if (B < 0.0) {
        sq = -sq;
    }
    double q = -0.5 * (B + sq);
    if (q == 0.0) { // only when C is exactly 0
        return {0.0, -B / A};
    }
    return {q / A, C / q};
}

inline double firstQuadRoot(double A, double B, double C) {
    vector2D r = quadRoots(A, B, C);
    return firstRoot(r.x, r.y);
}

inline double firstRootN(Roots r) {
    double t = MAGIC_NO_COLLISION;
    for (int i = 0; i < r.n; ++i) {
        if (r.t[i] > deadTime && r.t[i] < t) t = r.t[i];
    }
    return t;
}


// a polynomial of degree four or less, c0 + c1 t + ... + c4 t^4. a small
// struct passed by value stays in registers on the GPU, where an indexed
// array of coefficients would not
struct Poly4 {
    double c0, c1, c2, c3, c4;
};

// value and slope, by Horner's rule
inline double polyValue(Poly4 p, double t, double *slope) {
    *slope = ((4.0 * p.c4 * t + 3.0 * p.c3) * t + 2.0 * p.c2) * t + p.c1;
    return (((p.c4 * t + p.c3) * t + p.c2) * t + p.c1) * t + p.c0;
}

// the root of p in [lo, hi], an interval over which p is monotonic and changes
// sign, flo being its value at lo. Newton's method from the midpoint, falling
// back to bisection whenever a step would leave the bracket, so it converges
// from any start and can only land on a sign change
inline double bracketedRoot(Poly4 p, double lo, double hi, double flo) {
    double slope;
    bool negLo = flo < 0.0;
    double t = 0.5 * (lo + hi);
    for (int k = 0; k < 2100; ++k) {
        double f = polyValue(p, t, &slope);
        if (f == 0.0) return t;
        if ((f < 0.0) == negLo) lo = t; else hi = t; // keep the sign change inside
        double next = (slope != 0.0) ? t - f / slope : 0.5 * (lo + hi);
        if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
        if (!(next > lo && next < hi)) return t;  // lo and hi are adjacent doubles
        if (fabs(next - t) <= 1e-15 * fabs(next)) return next;
        t = next;
    }
    return t;
}

// a point beyond x, in direction dir = +-1, where p has the sign it takes at
// that infinity (negative when neg): steps double until the sign is reached.
// the value there is left in *f
inline double reachSign(Poly4 p, double x, double dir, bool neg, double *f) {
    double slope, step = fabs(x) > 1.0 ? fabs(x) : 1.0;
    double y = x + dir * step;
    for (int k = 0; k < 2100; ++k) {
        *f = polyValue(p, y, &slope);
        if (*f != 0.0 && (*f < 0.0) == neg) return y;
        step *= 2.0;
        y = x + dir * step;
    }
    return y;
}

// the real roots of p (degree d, its leading coefficient nonzero) that are
// larger than after, ascending, at most cap, given the ascending real roots
// x[0..nx) of its derivative. between two consecutive critical points p is
// monotonic, so each interval holds a root exactly when p changes sign across
// it. every interval is checked, and a root is reported only where the computed
// p changes sign, never from an estimate.
// a value of exactly zero at a critical point is a multiple root, reported once
inline int rootsBetween(Poly4 p, int d, const double *x, int nx, double after,
                        double *out, int cap) {
    double slope, fx[3];
    for (int i = 0; i < nx; ++i) fx[i] = polyValue(p, x[i], &slope);
    bool negRight = ((d == 4) ? p.c4 : p.c3) < 0.0;      // sign at +infinity
    bool negLeft = (d % 2 == 0) ? negRight : !negRight; // and at -infinity
    int n = 0;
    for (int i = 0; i <= nx && n < cap; ++i) {
        if (i < nx && x[i] <= after) continue;   // the whole interval lies before
        double fL = (i == 0) ? (negLeft ? -1.0 : 1.0) : fx[i - 1];
        double fR = (i == nx) ? (negRight ? -1.0 : 1.0) : fx[i];
        if (fL != 0.0 && fR != 0.0 && (fL < 0.0) != (fR < 0.0)) {
            double lo, hi;
            if (i == 0) lo = reachSign(p, nx > 0 ? x[0] : 0.0, -1.0, negLeft, &fL);
            else lo = x[i - 1];
            if (i == nx) hi = reachSign(p, nx > 0 ? x[nx - 1] : 0.0, 1.0, negRight, &fR);
            else hi = x[i];
            double t = bracketedRoot(p, lo, hi, fL);
            if (t > after) out[n++] = t;
        }
        if (i < nx && fx[i] == 0.0 && n < cap) out[n++] = x[i];
    }
    return n;
}

// the real roots of A4 t^4 + A3 t^3 + A2 t^2 + A1 t + A0 that come after
// deadTime, ascending: the only ones a hit time can use. the quartic is
// monotonic between the roots of its derivative, a cubic, which is monotonic
// between the roots of its own derivative, a quadratic solved in closed form:
// the roots of each level bracket those of the level above, so every root is
// located inside a bracket, never estimated
inline Roots quartRoots(double A4, double A3, double A2, double A1, double A0) {
    Roots out = {{0.0, 0.0, 0.0, 0.0}, 0};

    // A4 = (g^2/4)(sin^2/a^2 + cos^2/b^2) is exactly zero when g == 0, and only
    // then: the trajectory is a straight line, and A3 vanishes with it. a tiny
    // but nonzero g still bends the flight back eventually, so it is solved in full
    if (A4 == 0.0) {
        vector2D r = quadRoots(A2, A1, A0);
        if (r.x != MAGIC_NO_COLLISION) out.t[out.n++] = r.x;
        if (r.y != MAGIC_NO_COLLISION) out.t[out.n++] = r.y;
        return out;
    }

    Poly4 quartic = {A0, A1, A2, A3, A4};
    Poly4 cubic = {A1, 2.0 * A2, 3.0 * A3, 4.0 * A4, 0.0};

    // inflection points: the roots of the second derivative
    double infl[2];
    int ni = 0;
    vector2D r2 = quadRoots(12.0 * A4, 6.0 * A3, 2.0 * A2);
    if (r2.x != MAGIC_NO_COLLISION) infl[ni++] = r2.x;
    if (r2.y != MAGIC_NO_COLLISION) infl[ni++] = r2.y;
    if (ni == 2 && infl[1] < infl[0]) {
        double tmp = infl[0];
        infl[0] = infl[1];
        infl[1] = tmp;
    }

    // every critical point is needed, past ones included: they bound the
    // intervals that reach into the future
    double crit[3];
    int nc = rootsBetween(cubic, 3, infl, ni, -INFINITY, crit, 3);
    out.n = rootsBetween(quartic, 4, crit, nc, deadTime, out.t, 4);
    return out;
}

// ------------------------------------------------------------------ 2. makers

// scale the coefficients so that (a, b) is the unit normal
inline Object makeLine(double a, double b, double c) {
    double len = sqrt(a * a + b * b);
    return {LINE, REFLECT, {a / len, b / len, c / len}, {}};
}

inline Object makeLineSegment(double x0, double y0, double x1, double y1) {
    return {LINE_SEGMENT, REFLECT, {x0, y0, x1, y1}, {}};
}

inline Object makeEllipse(double x0, double y0, double a, double b, double theta) {
    return {ELLIPSE, REFLECT, {x0, y0, a, b, theta, cos(theta), sin(theta)}, {}};
}

// counterclockwise from phi0 to phi1, in the ellipse's own frame. the endpoint
// cosines and sines are taken here, on the host, so onArc can decide containment
// with sign tests alone: a transcendental on the device would not match the
// host's bit for bit
inline Object makeEllipseArc(double x0, double y0, double a, double b, double theta,
                            double phi0, double phi1) {
    return {ELLIPSE_ARC, REFLECT, {x0, y0, a, b, theta, cos(theta), sin(theta),
                                  phi0, phi1, cos(phi0), sin(phi0), cos(phi1), sin(phi1)},
            {}};
}

// ------------------------------------------------------------------ 3. lines

// substitute the parabola into a x + b y + c: A t^2 + B t + C
inline vector2D lineRoots(double a, double b, double c, state s) {
    return quadRoots(-0.5 * g * b,
                     a * s.vx + b * s.vy,
                     a * s.x + b * s.y + c);
}

inline double lineHitTime(Object o, state s) {
    vector2D r = lineRoots(o.p[0], o.p[1], o.p[2], s);
    return firstRoot(r.x, r.y);
}

inline vector2D lineNormal(Object o, state s) {
    return {o.p[0], o.p[1]};
}

// move by minus the signed distance, along the normal
inline state lineSnap(Object o, state s) {
    double d = o.p[0] * s.x + o.p[1] * s.y + o.p[2];
    return {s.x - d * o.p[0], s.y - d * o.p[1], s.vx, s.vy};
}

// --------------------------------------------------------------- 4. segments

inline Object segmentLine(Object o) {
    double x0 = o.p[0], y0 = o.p[1], x1 = o.p[2], y1 = o.p[3];
    return makeLine(y0 - y1, x1 - x0, x0 * y1 - x1 * y0);
}

// u runs 0 at one endpoint, 1 at the other
inline bool onSegment(Object o, double t, state s) {
    state at = trajectory(s, t);
    double dx = o.p[2] - o.p[0], dy = o.p[3] - o.p[1];
    double u = ((at.x - o.p[0]) * dx + (at.y - o.p[1]) * dy) / (dx * dx + dy * dy);
    return u >= 0.0 && u <= 1.0;
}

// both roots are tested: the earlier one can sail past an endpoint while the
// later one lands
inline double segmentHitTime(Object o, state s) {
    Object l = segmentLine(o);
    vector2D r = lineRoots(l.p[0], l.p[1], l.p[2], s);
    if (!onSegment(o, r.x, s)) r.x = MAGIC_NO_COLLISION;
    if (!onSegment(o, r.y, s)) r.y = MAGIC_NO_COLLISION;
    return firstRoot(r.x, r.y);
}

// no orientation to pick: bounce is quadratic in n, so a segment hits both ways
inline vector2D segmentNormal(Object o, state s) {
    return lineNormal(segmentLine(o), s);
}

inline state segmentSnap(Object o, state s) {
    return lineSnap(segmentLine(o), s);
}

// --------------------------------------------------------------- 4a. portals

// the two signs act on the velocity alone. WHERE the particle comes out is fixed
// by the frames, and so by the order the partner's endpoints were written in:
// naming it makeLineSegment(8,8, 8,2) rather than (8,2, 8,8) maps end for end.
// that also turns the tangent over, and the normal with it, so pass
// flipNormal = -1 alongside to keep the original exit face.
//
// the exit takes its origin and direction from the partner but its LENGTH from
// here, so a mismatched pair cannot stretch the particle along the surface
inline Object asPortal(Object o, Object partner, double flipNormal,
                       double flipTangent = 1.0) {
    double ax = o.p[2] - o.p[0], ay = o.p[3] - o.p[1];
    double len = sqrt(ax * ax + ay * ay);

    double bx = partner.p[2] - partner.p[0], by = partner.p[3] - partner.p[1];
    double blen = sqrt(bx * bx + by * by);
    double tx = bx / blen, ty = by / blen;
    double ox = partner.p[0], oy = partner.p[1];

    o.response = PORTAL;
    o.q[0] = o.p[0]; o.q[1] = o.p[1]; o.q[2] = o.p[2]; o.q[3] = o.p[3];
    // a whole length out, not one unit: frameFromPoints normalises whatever it is
    // handed, and normalising a unit vector is not the identity in floating point
    o.q[4] = ox;                o.q[5] = oy;
    o.q[6] = ox + len * tx;     o.q[7] = oy + len * ty;
    o.q[8] = flipNormal;
    o.q[9] = flipTangent;
    return o;
}

// a basin ends the run instead of responding to it. any shape can become one --
// the geometry is untouched, so it is hit-tested by exactly the same hitTime as
// the wall it would otherwise have been.
//
// the label is what a caller gets back, and several shapes may share one: a
// basin is a set of surfaces, not a surface. 0 is not usable here, it is what a
// particle that reached the time limit reports
inline Object asBasin(Object o, double label) {
    o.response = ABSORB;
    o.q[0] = label;
    return o;
}

// --------------------------------------------------------------- 5. ellipses

// everything here works in the ellipse's own frame, where the boundary is just
// u^2/a^2 + v^2/b^2 = 1

inline vector2D ellipseLocal(Object o, double x, double y) {
    double c = o.p[5], s = o.p[6];
    double dx = x - o.p[0], dy = y - o.p[1];
    return {c * dx + s * dy, -s * dx + c * dy};
}

inline vector2D ellipseToWorld(Object o, double u, double v, bool isPoint) {
    double c = o.p[5], s = o.p[6];
    return {c * u - s * v + (isPoint ? o.p[0] : 0.0),
            s * u + c * v + (isPoint ? o.p[1] : 0.0)};
}

// u and v are both quadratic in t, so their squares are quartic. A0 is negative
// inside the ellipse and positive outside, which is why neither case needs its
// own branch. kept apart from ellipseHitTime so an arc can see every root
// after deadTime, which is all quartRoots returns
inline Roots ellipseRoots(Object o, state s) {
    double c = o.p[5], sn = o.p[6];
    double ia = 1.0 / (o.p[2] * o.p[2]), ib = 1.0 / (o.p[3] * o.p[3]);
    double dx = s.x - o.p[0], dy = s.y - o.p[1];

    double u0 = c * dx + sn * dy, u1 = c * s.vx + sn * s.vy, u2 = -0.5 * g * sn;
    double v0 = -sn * dx + c * dy, v1 = -sn * s.vx + c * s.vy, v2 = -0.5 * g * c;

    return quartRoots(ia * u2 * u2 + ib * v2 * v2,
                      2.0 * (ia * u1 * u2 + ib * v1 * v2),
                      ia * (u1 * u1 + 2.0 * u0 * u2) + ib * (v1 * v1 + 2.0 * v0 * v2),
                      2.0 * (ia * u0 * u1 + ib * v0 * v1),
                      ia * u0 * u0 + ib * v0 * v0 - 1.0);
}

inline double ellipseHitTime(Object o, state s) {
    return firstRootN(ellipseRoots(o, s));
}

// the gradient of u^2/a^2 + v^2/b^2 points straight out of the surface
inline vector2D ellipseNormal(Object o, state s) {
    vector2D l = ellipseLocal(o, s.x, s.y);
    double nu = l.x / (o.p[2] * o.p[2]), nv = l.y / (o.p[3] * o.p[3]);
    double len = sqrt(nu * nu + nv * nv);
    if (len == 0.0) return {0.0, 0.0}; // dead centre
    return ellipseToWorld(o, nu / len, nv / len, false);
}

// back onto the boundary along its own radius. not the nearest point on the
// ellipse -- that is another quartic -- but exactly on the curve, and differing
// only to second order
inline state ellipseSnap(Object o, state s) {
    vector2D l = ellipseLocal(o, s.x, s.y);
    double ia = 1.0 / (o.p[2] * o.p[2]), ib = 1.0 / (o.p[3] * o.p[3]);
    double d = sqrt(ia * l.x * l.x + ib * l.y * l.y);
    if (d == 0.0) return s;
    vector2D w = ellipseToWorld(o, l.x / d, l.y / d, true);
    return {w.x, w.y, s.vx, s.vy};
}

// ----------------------------------------------------------- 6. ellipse arcs

// on the ellipse (u/a, v/b) is already (cos phi, sin phi), so the parametric
// angle needs no inverse trig. left unnormalised on purpose: a positive scale
// cannot flip the sign of a cross product, and only signs are read below.
// atan2 would say the same in one line and is deliberately not used -- it is not
// correctly rounded, and the device's disagrees with glibc's in the last bits
inline bool onArc(Object o, double t, state s) {
    state at = trajectory(s, t);
    vector2D l = ellipseLocal(o, at.x, at.y);
    double dx = l.x / o.p[2], dy = l.y / o.p[3];

    double c0 = o.p[9], s0 = o.p[10], c1 = o.p[11], s1 = o.p[12];
    double cr0 = c0 * dy - s0 * dx; // sin(phi  - phi0)
    double cr1 = dx * s1 - dy * c1; // sin(phi1 - phi)
    double crS = c0 * s1 - s0 * c1; // sin(phi1 - phi0)

    // a sweep of half a turn or less is convex, so the point must be past phi0 and
    // short of phi1 at once. a longer sweep is the complement of the short gap left
    // over, and being past either end is enough
    if (crS >= 0.0) return cr0 >= 0.0 && cr1 >= 0.0;
    return cr0 >= 0.0 || cr1 >= 0.0;
}

// all the roots are tested, not just the earliest: a particle coming in through
// the open side misses the arc on the near root and lands on it on the far one
inline double ellipseArcHitTime(Object o, state s) {
    Roots r = ellipseRoots(o, s);
    int kept = 0;
    for (int i = 0; i < r.n; ++i) {
        if (onArc(o, r.t[i], s)) r.t[kept++] = r.t[i];
    }
    r.n = kept;
    return firstRootN(r);
}

inline vector2D ellipseArcNormal(Object o, state s) { return ellipseNormal(o, s); }

inline state ellipseArcSnap(Object o, state s) { return ellipseSnap(o, s); }

// --------------------------------------------------------------- 7. dispatch

inline double hitTime(Object o, state s) {
    switch (o.type) {
    case LINE: return lineHitTime(o, s);
    case LINE_SEGMENT: return segmentHitTime(o, s);
    case ELLIPSE: return ellipseHitTime(o, s);
    case ELLIPSE_ARC: return ellipseArcHitTime(o, s);
    }
    return MAGIC_NO_COLLISION;
}

inline vector2D normalAt(Object o, state s) {
    switch (o.type) {
    case LINE: return lineNormal(o, s);
    case LINE_SEGMENT: return segmentNormal(o, s);
    case ELLIPSE: return ellipseNormal(o, s);
    case ELLIPSE_ARC: return ellipseArcNormal(o, s);
    }
    return {0.0, 0.0};
}

// project the leftover rounding back onto the surface
inline state snapTo(Object o, state s) {
    switch (o.type) {
    case LINE: return lineSnap(o, s);
    case LINE_SEGMENT: return segmentSnap(o, s);
    case ELLIPSE: return ellipseSnap(o, s);
    case ELLIPSE_ARC: return ellipseArcSnap(o, s);
    }
    return s;
}

#ifdef _OPENMP
#pragma omp end declare target
#endif
