#pragma once

#include "types.h"
#include <cmath>


/*
Shapes do include
"Full shapes":
Elipse, Hyperbola, Parabola

"Fractioned shapes"
CircleArc, ElipseArc, HyperbolaArc, ParabolaArc, LineSegments.


*/

// LINE: a, b, c of a x + b y + c = 0, (a,b) kept unit

// LINE_SEGMENT: x0, y0, x1, y1, rest is calculated on the fly

// ELIPSE: x0, y0, a, b, theta, the elipse can be rotated.
//         makeElipse appends cos(theta), sin(theta): they never change after
//         construction, and the hot path would otherwise pay for them twice a bounce

// ELIPSE_ARC: x0, y0, a, b, theta, phi0, phi1. phi0/1 are the beginning/end of the arc in radians, follwing the trigonometry configuration



enum ObjectType { LINE = 0, 
                  LINE_SEGMENT = 1,
                  ELIPSE = 2}; // Line for now, but will include Cicle, LineSegment, 
struct Object {
    int type;
    double p[7]; // LINE:         a, b, c of a x + b y + c = 0, (a,b) kept unit
                 // LINE_SEGMENT: x0, y0, x1, y1
                 // ELIPSE:       x0, y0, a, b, theta, cos(theta), sin(theta)
};

#ifdef _OPENMP
#pragma omp begin declare target
#endif

// ------------------------------------------------------------- 0. ballistics

// ballistic trajectory state
inline state trajectory(state s, double t) {
    return {s.x + s.vx * t,
            s.y + s.vy * t - 0.5 * g * t * t,
            s.vx,
            s.vy - g * t};
}

// ---------------------------------------------------------- 1. root finding

// just check the smaller root
inline double firstRoot(double t1, double t2) {
    double t = MAGIC_NO_COLLISION;
    if (t1 > deadTime && t1 < t) t = t1;
    if (t2 > deadTime && t2 < t) t = t2;
    return t;
}

// Find quadratic roots of A t^2 + B t + C, returning MAGIC_NO_COLLISION if none exist
inline vector2D quadRoots(double A, double B, double C) {

    // an early "return {CONST, CONST};" in device code and hands back zeros,
    // which reads as a collision at t = 0 instead of no collision at all
    vector2D none = {MAGIC_NO_COLLISION, MAGIC_NO_COLLISION};

    // special cases
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
    // ill conditioned, which is far coarser than deadTime and would let a
    // particle slip through a wall. two Newton steps restore full precision;
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

// scale the coefficients so that (a, b) is the unit normal
inline Object makeLine(double a, double b, double c) {
    double len = sqrt(a * a + b * b);
    return {LINE, {a / len, b / len, c / len}};
}

// the endpoints are the whole story: the line through them is rebuilt on the fly
inline Object makeLineSegment(double x0, double y0, double x1, double y1) {
    return {LINE_SEGMENT, {x0, y0, x1, y1}};
}

// theta rotates the ellipse counterclockwise. its cosine and sine are stored
inline Object makeElipse(double x0, double y0, double a, double b, double theta) {
    return {ELIPSE, {x0, y0, a, b, theta, cos(theta), sin(theta)}};
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

// Jump to segment
inline bool onSegment(Object o, double t, state s) {
    state at = trajectory(s, t);
    double dx = o.p[2] - o.p[0], dy = o.p[3] - o.p[1];
    double u = ((at.x - o.p[0]) * dx + (at.y - o.p[1]) * dy) / (dx * dx + dy * dy);
    return u >= 0.0 && u <= 1.0;
}

// the line's own roots, minus the ones that land off the ends. Both are needed:
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
inline double elipseHitTime(Object o, state s) {
    double c = o.p[5], sn = o.p[6];
    double ia = 1.0 / (o.p[2] * o.p[2]), ib = 1.0 / (o.p[3] * o.p[3]);
    double dx = s.x - o.p[0], dy = s.y - o.p[1];

    double u0 = c * dx + sn * dy, u1 = c * s.vx + sn * s.vy, u2 = -0.5 * g * sn;
    double v0 = -sn * dx + c * dy, v1 = -sn * s.vx + c * s.vy, v2 = -0.5 * g * c;

    Roots r = quartRoots(ia * u2 * u2 + ib * v2 * v2,
                         2.0 * (ia * u1 * u2 + ib * v1 * v2),
                         ia * (u1 * u1 + 2.0 * u0 * u2) + ib * (v1 * v1 + 2.0 * v0 * v2),
                         2.0 * (ia * u0 * u1 + ib * v0 * v1),
                         ia * u0 * u0 + ib * v0 * v0 - 1.0);
    return firstRootN(r);
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

// --------------------------------------------------------------- 6. dispatch

// time at which the trajectory meets the object, MAGIC_NO_COLLISION if never
inline double hitTime(Object o, state s) {
    switch (o.type) {
    case LINE: return lineHitTime(o, s);
    case LINE_SEGMENT: return segmentHitTime(o, s);
    case ELIPSE: return elipseHitTime(o, s);
    }
    return MAGIC_NO_COLLISION;
}

// unit normal at the contact point
inline vector2D normalAt(Object o, state s) {
    switch (o.type) {
    case LINE: return lineNormal(o, s);
    case LINE_SEGMENT: return segmentNormal(o, s);
    case ELIPSE: return elipseNormal(o, s);
    }
    return {0.0, 0.0};
}

// project the leftover rounding back onto the surface
inline state snapTo(Object o, state s) {
    switch (o.type) {
    case LINE: return lineSnap(o, s);
    case LINE_SEGMENT: return segmentSnap(o, s);
    case ELIPSE: return elipseSnap(o, s);
    }
    return s;
}

#ifdef _OPENMP
#pragma omp end declare target
#endif
