#pragma once

#include "types.h"
#include <cmath>


// the shape says where contact happens, the response says what contact does
enum ObjectType { LINE = 0,
                  LINE_SEGMENT = 1,
                  ELIPSE = 2,
                  ELIPSE_ARC = 3};

enum Response { REFLECT = 0,   // mirror the velocity about the normal
                PORTAL = 1};   // move the particle to the partner named in q

struct Object {
    int type;
    int response;
    double p[13]; // LINE:         a, b, c of a x + b y + c = 0, (a,b) kept unit
                  // LINE_SEGMENT: x0, y0, x1, y1
                  // ELIPSE:       x0, y0, a, b, theta, cos(theta), sin(theta)
                  // ELIPSE_ARC:   the seven above, then
                  //               phi0, phi1, cos(phi0), sin(phi0), cos(phi1), sin(phi1)
                  //
                  // an arc's first seven slots are byte for byte an ELIPSE, which is
                  // why every elipse* function below reads an arc without conversion

    double q[10]; // PORTAL: a0x, a0y, a1x, a1y  this portal's own frame
                  //         b0x, b0y, b1x, b1y  the frame it leads to
                  //         flipNormal   +1 carry through, -1 leave by the front face
                  //         flipTangent  +1 same sense,    -1 mirrored about the origin
                  //
                  // the partner is held as coordinates, never as an index into the
                  // object array: nothing to dereference on the device, and the
                  // objects can be listed in any order
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

// the smaller of two candidate times that is still ahead of us
inline double firstRoot(double t1, double t2) {
    double t = MAGIC_NO_COLLISION;
    if (t1 > deadTime && t1 < t) t = t1;
    if (t2 > deadTime && t2 < t) t = t2;
    return t;
}

// Find quadratic roots of A t^2 + B t + C, returning MAGIC_NO_COLLISION if none exist
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

    // stable form: compute q with the sign of B, so B + sq never cancels;
    // one root is q/A, the other C/q
    double sq = sqrt(disc);
    if (B < 0.0) {
        sq = -sq;
    }
    double q = -0.5 * (B + sq);
    if (q == 0.0) { // only when C is exactly 0: the roots are 0 and -B/A
        return {0.0, -B / A};
    }
    return {q / A, C / q};
}

// smallest root > deadTime of A t^2 + B t + C
inline double firstQuadRoot(double A, double B, double C) {
    vector2D r = quadRoots(A, B, C);
    return firstRoot(r.x, r.y);
}

// the same idea as firstRoot, for the variable-length list a quartic returns
inline double firstRootN(Roots r) {
    double t = MAGIC_NO_COLLISION;
    for (int i = 0; i < r.n; ++i) {
        if (r.t[i] > deadTime && r.t[i] < t) t = r.t[i];
    }
    return t;
}


// a positive real root of the monic cubic m^3 + B m^2 + C m + D, for a cubic
// that is negative at zero. Ferrari's resolvent always is, so a root above zero
// is guaranteed to exist and that is the only caller.
// Cauchy's bound sits above every root, so starting there and walking down finds
// the largest one first, which keeps the q/s division in Ferrari well scaled.
// Newton does the walking and bisection catches it whenever it leaves the
// bracket, which makes convergence unconditional
inline double cubicRootAboveZero(double B, double C, double D) {
    double lo = 0.0; // f(0) = D < 0 by precondition
    double hi = 1.0 + fabs(B);
    if (fabs(C) > hi - 1.0) hi = 1.0 + fabs(C);
    if (fabs(D) > hi - 1.0) hi = 1.0 + fabs(D);

    double m = hi; // above every root, where the cubic is positive
    for (int k = 0; k < 200; ++k) {
        double f = ((m + B) * m + C) * m + D;
        if (f > 0.0) hi = m; else lo = m; // keep f(lo) <= 0 <= f(hi)

        double df = (3.0 * m + 2.0 * B) * m + C;
        double next = (df != 0.0) ? m - f / df : 0.5 * (lo + hi);
        if (!(next > lo && next < hi)) next = 0.5 * (lo + hi); // also catches NaN

        if (fabs(next - m) <= 1e-17 * fabs(m)) return next;
        m = next;
    }
    return m;
}

// real roots of A4 t^4 + A3 t^3 + A2 t^2 + A1 t + A0, by Ferrari's method
inline Roots quartRoots(double A4, double A3, double A2, double A1, double A0) {
    Roots out = {{0.0, 0.0, 0.0, 0.0}, 0};

    double big = fabs(A3);
    if (fabs(A2) > big) big = fabs(A2);
    if (fabs(A1) > big) big = fabs(A1);
    if (fabs(A0) > big) big = fabs(A0);
    if (fabs(A4) <= 1e-14 * big) { // g == 0: the trajectory is a straight line
        vector2D r = quadRoots(A2, A1, A0);
        if (r.x != MAGIC_NO_COLLISION) out.t[out.n++] = r.x;
        if (r.y != MAGIC_NO_COLLISION) out.t[out.n++] = r.y;
        return out;
    }

    // depress with t = z - a3/4: z^4 + p z^2 + q z + r
    double a3 = A3 / A4, a2 = A2 / A4, a1 = A1 / A4, a0 = A0 / A4;
    double p = a2 - 3.0 * a3 * a3 / 8.0;
    double q = a1 - a2 * a3 / 2.0 + a3 * a3 * a3 / 8.0;
    double r = a0 - a1 * a3 / 4.0 + a2 * a3 * a3 / 16.0 - 3.0 * a3 * a3 * a3 * a3 / 256.0;
    double shift = a3 / 4.0;

    double qScale = fabs(p) > fabs(r) ? fabs(p) : fabs(r);
    if (qScale < 1.0) qScale = 1.0;

    if (fabs(q) <= 1e-14 * qScale) { // biquadratic: solve for z^2, then take roots
        vector2D z2 = quadRoots(1.0, p, r);
        double c2[2] = {z2.x, z2.y};
        for (int i = 0; i < 2; ++i) {
            if (c2[i] != MAGIC_NO_COLLISION && c2[i] >= 0.0) {
                double z = sqrt(c2[i]);
                out.t[out.n++] = z - shift;
                out.t[out.n++] = -z - shift;
            }
        }
    } else {
        // factor into (z^2 + s z + u)(z^2 - s z + w). matching coefficients gives
        // u + w = p + s^2, w - u = q/s, u w = r, and eliminating u and w leaves
        // this cubic in m = s^2. at m = 0 it equals -q^2 < 0 and grows without
        // bound, so a positive root always exists
        double m = cubicRootAboveZero(2.0 * p, p * p - 4.0 * r, -q * q);
        if (m <= 0.0) return out;

        double s = sqrt(m);
        double u = ((p + m) - q / s) / 2.0;
        double w = ((p + m) + q / s) / 2.0;

        // quadRoots again rather than a second copy of its cancellation-safe form
        vector2D r1 = quadRoots(1.0, s, u);
        vector2D r2 = quadRoots(1.0, -s, w);
        double z[4] = {r1.x, r1.y, r2.x, r2.y};
        for (int i = 0; i < 4; ++i) {
            if (z[i] != MAGIC_NO_COLLISION) out.t[out.n++] = z[i] - shift;
        }
    }

    // Ferrari alone lands only within ~1e-5 of the root once the resolvent is
    // ill conditioned, far coarser than deadTime and enough to let a particle
    // slip through a wall. two Newton steps reach machine precision; a third
    // measurably buys nothing
    int kept = 0;
    for (int i = 0; i < out.n; ++i) {
        double t = out.t[i];
        for (int k = 0; k < 2; ++k) {
            double F = (((A4 * t + A3) * t + A2) * t + A1) * t + A0;
            double dF = ((4.0 * A4 * t + 3.0 * A3) * t + 2.0 * A2) * t + A1;
            if (dF == 0.0) break; // a double root: already as good as it gets
            t -= F / dF;
        }

        // and then check it really is a root. on a near miss the factorisation
        // can hand back a value that never crossed zero, and Newton merely parks
        // it near a minimum of F; accepting that invents a collision out of thin
        // air and teleports the particle onto the surface. the residual is the
        // only thing that tells a true crossing from a graze
        double F = (((A4 * t + A3) * t + A2) * t + A1) * t + A0;
        double m = fabs(t);
        double scale = (((fabs(A4) * m + fabs(A3)) * m + fabs(A2)) * m + fabs(A1)) * m + fabs(A0);
        if (scale < 1.0) scale = 1.0;
        if (fabs(F) <= 1e-10 * scale) out.t[kept++] = t;
    }
    out.n = kept;
    return out;
}

// ------------------------------------------------------------------ 2. makers

// each maker names its response rather than leaning on REFLECT being zero

// scale the coefficients so that (a, b) is the unit normal
inline Object makeLine(double a, double b, double c) {
    double len = sqrt(a * a + b * b);
    return {LINE, REFLECT, {a / len, b / len, c / len}, {}};
}

// the endpoints are the whole story: the line through them is rebuilt on the fly
inline Object makeLineSegment(double x0, double y0, double x1, double y1) {
    return {LINE_SEGMENT, REFLECT, {x0, y0, x1, y1}, {}};
}

// theta rotates the ellipse counterclockwise. its cosine and sine are stored
inline Object makeElipse(double x0, double y0, double a, double b, double theta) {
    return {ELIPSE, REFLECT, {x0, y0, a, b, theta, cos(theta), sin(theta)}, {}};
}

// the arc runs counterclockwise from phi0 to phi1, in the ellipse's own frame.
// the endpoint directions are turned into cosines and sines here, on the host,
// so that onArc can decide containment with sign tests alone: an atan2 in the
// hot path would be a transcendental on the device, and those do not match the
// host's bit for bit
inline Object makeElipseArc(double x0, double y0, double a, double b, double theta,
                            double phi0, double phi1) {
    return {ELIPSE_ARC, REFLECT, {x0, y0, a, b, theta, cos(theta), sin(theta),
                                  phi0, phi1, cos(phi0), sin(phi0), cos(phi1), sin(phi1)},
            {}};
}

// ------------------------------------------------------------------ 3. lines

// substitute the parabola into a x + b y + c: A t^2 + B t + C.
inline vector2D lineRoots(double a, double b, double c, state s) {
    return quadRoots(-0.5 * g * b,
                     a * s.vx + b * s.vy,
                     a * s.x + b * s.y + c);
}

inline double lineHitTime(Object o, state s) {
    vector2D r = lineRoots(o.p[0], o.p[1], o.p[2], s);
    return firstRoot(r.x, r.y);
}

// same unit normal (a, b) everywhere (a circle's would depend on s)
inline vector2D lineNormal(Object o, state s) {
    return {o.p[0], o.p[1]};
}

// land exactly on the line: move by minus the signed distance, along the normal
inline state lineSnap(Object o, state s) {
    double d = o.p[0] * s.x + o.p[1] * s.y + o.p[2];
    return {s.x - d * o.p[0], s.y - d * o.p[1], s.vx, s.vy};
}

// --------------------------------------------------------------- 4. segments

// the infinite line through the two endpoints, (a, b) made unit by makeLine
inline Object segmentLine(Object o) {
    double x0 = o.p[0], y0 = o.p[1], x1 = o.p[2], y1 = o.p[3];
    return makeLine(y0 - y1, x1 - x0, x0 * y1 - x1 * y0);
}

// is the impact point between the endpoints? u runs 0 at one, 1 at the other
inline bool onSegment(Object o, double t, state s) {
    state at = trajectory(s, t);
    double dx = o.p[2] - o.p[0], dy = o.p[3] - o.p[1];
    double u = ((at.x - o.p[0]) * dx + (at.y - o.p[1]) * dy) / (dx * dx + dy * dy);
    return u >= 0.0 && u <= 1.0;
}

// the line's own roots, minus the ones landing off the ends. both are needed:
// the earlier root can sail past an endpoint while the later one lands
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

// a portal moves the particle to its partner instead of reflecting it. the
// partner is named as an Object rather than having its coordinates respelled,
// so the two can never drift apart.
//
// the two signs act on the velocity alone: flipNormal picks which face the
// particle leaves by, flipTangent which way it runs along the partner afterwards.
//
// WHERE it comes out is not theirs to say. that is fixed by the frames, and so by
// the order the partner's endpoints were written in: name the partner
// makeLineSegment(8,8, 8,2) rather than makeLineSegment(8,2, 8,8) and the mapping
// runs end for end instead of end to end. reversing the order turns the tangent
// over, and the normal is the tangent turned a quarter turn, so the exit face
// swaps with it -- pass flipNormal = -1 alongside to keep the original side

// a segment is oriented by its own endpoints and needs no anchor. the exit takes
// its origin and direction from the partner but its LENGTH from here, so a pair
// of unequal size cannot stretch the particle along the surface
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
    // the far point is placed a whole length away, not one unit: frameFromPoints
    // normalises whatever it is handed, and normalising a unit vector is not the
    // identity in floating point. keeping the length here makes the stored frame
    // the segment's own, to the last bit
    o.q[4] = ox;                o.q[5] = oy;
    o.q[6] = ox + len * tx;     o.q[7] = oy + len * ty;
    o.q[8] = flipNormal;
    o.q[9] = flipTangent;
    return o;
}

// --------------------------------------------------------------- 5. elipses

// an ellipse is the first shape whose own equation is quadratic, so feeding the
// parabola into it leaves a quartic in t rather than the quadratic every shape
// above reduces to. everything here works in the ellipse's own frame, where the
// boundary is just u^2/a^2 + v^2/b^2 = 1

// world point -> the frame in which the ellipse is axis aligned and centred
inline vector2D elipseLocal(Object o, double x, double y) {
    double c = o.p[5], s = o.p[6];
    double dx = x - o.p[0], dy = y - o.p[1];
    return {c * dx + s * dy, -s * dx + c * dy};
}

// the reverse trip, for a direction (no centre offset) or a point (with one)
inline vector2D elipseToWorld(Object o, double u, double v, bool isPoint) {
    double c = o.p[5], s = o.p[6];
    return {c * u - s * v + (isPoint ? o.p[0] : 0.0),
            s * u + c * v + (isPoint ? o.p[1] : 0.0)};
}

// substitute the parabola into u^2/a^2 + v^2/b^2 - 1. both u and v are quadratic
// in t, so their squares are quartic. A0 is where the particle starts: negative
// inside the ellipse, positive outside, which is why neither case needs its own
// branch -- the roots come out the same way regardless
// kept apart from elipseHitTime so an arc can look at every root before one is
// chosen, exactly as lineRoots is kept apart for the segment's sake
inline Roots elipseRoots(Object o, state s) {
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

inline double elipseHitTime(Object o, state s) {
    return firstRootN(elipseRoots(o, s));
}

// the gradient of u^2/a^2 + v^2/b^2 points straight out of the surface
inline vector2D elipseNormal(Object o, state s) {
    vector2D l = elipseLocal(o, s.x, s.y);
    double nu = l.x / (o.p[2] * o.p[2]), nv = l.y / (o.p[3] * o.p[3]);
    double len = sqrt(nu * nu + nv * nv);
    if (len == 0.0) return {0.0, 0.0}; // dead centre: no surface to speak of
    return elipseToWorld(o, nu / len, nv / len, false);
}

// pull the point back onto the boundary along its own radius. that is not the
// nearest point on the ellipse -- finding that is another quartic -- but it lands
// exactly on the curve and differs only to second order, which is all this is for
inline state elipseSnap(Object o, state s) {
    vector2D l = elipseLocal(o, s.x, s.y);
    double ia = 1.0 / (o.p[2] * o.p[2]), ib = 1.0 / (o.p[3] * o.p[3]);
    double d = sqrt(ia * l.x * l.x + ib * l.y * l.y);
    if (d == 0.0) return s;
    vector2D w = elipseToWorld(o, l.x / d, l.y / d, true);
    return {w.x, w.y, s.vx, s.vy};
}

// ----------------------------------------------------------- 6. elipse arcs

// an arc is the ellipse carrying it, plus an angular bound: the roots come from
// the ellipse untouched and only the ones landing between phi0 and phi1 survive,
// the same division of labour a segment has with its line

// is the impact point inside the arc's span?
//
// on the ellipse u^2/a^2 + v^2/b^2 = 1, so (u/a, v/b) is already (cos phi, sin phi)
// and the parametric angle needs no inverse trig to reach. it is left unnormalised
// on purpose: scaling by a positive number cannot flip the sign of a cross product,
// and only signs are read below.
//
// atan2 would say the same thing in one line, and is deliberately not used: it is
// not correctly rounded, the device's version disagrees with glibc's in the last
// bits, and that alone would cost the CPU and GPU their bit-for-bit agreement
inline bool onArc(Object o, double t, state s) {
    state at = trajectory(s, t);
    vector2D l = elipseLocal(o, at.x, at.y);
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

// the ellipse's roots, minus the ones landing off the ends. all of them have to be
// tested, not just the earliest: a particle coming in through the open side misses
// the arc on the near root and lands on it on the far one
inline double elipseArcHitTime(Object o, state s) {
    Roots r = elipseRoots(o, s);
    int kept = 0;
    for (int i = 0; i < r.n; ++i) {
        if (onArc(o, r.t[i], s)) r.t[kept++] = r.t[i];
    }
    r.n = kept;
    return firstRootN(r);
}

// no conversion needed: an arc's first seven parameters are an ellipse already
inline vector2D elipseArcNormal(Object o, state s) { return elipseNormal(o, s); }

inline state elipseArcSnap(Object o, state s) { return elipseSnap(o, s); }

// --------------------------------------------------------------- 7. dispatch

// time at which the trajectory meets the object, MAGIC_NO_COLLISION if never
inline double hitTime(Object o, state s) {
    switch (o.type) {
    case LINE: return lineHitTime(o, s);
    case LINE_SEGMENT: return segmentHitTime(o, s);
    case ELIPSE: return elipseHitTime(o, s);
    case ELIPSE_ARC: return elipseArcHitTime(o, s);
    }
    return MAGIC_NO_COLLISION;
}

// unit normal at the contact point
inline vector2D normalAt(Object o, state s) {
    switch (o.type) {
    case LINE: return lineNormal(o, s);
    case LINE_SEGMENT: return segmentNormal(o, s);
    case ELIPSE: return elipseNormal(o, s);
    case ELIPSE_ARC: return elipseArcNormal(o, s);
    }
    return {0.0, 0.0};
}

// project the leftover rounding back onto the surface
inline state snapTo(Object o, state s) {
    switch (o.type) {
    case LINE: return lineSnap(o, s);
    case LINE_SEGMENT: return segmentSnap(o, s);
    case ELIPSE: return elipseSnap(o, s);
    case ELIPSE_ARC: return elipseArcSnap(o, s);
    }
    return s;
}

#ifdef _OPENMP
#pragma omp end declare target
#endif
