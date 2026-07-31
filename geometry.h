#pragma once

#include "types.h"
#include <cmath>


/*
Shapes do include
"Full shapes":
Circle, Elipse, Hyperbola, Parabola

"Fractioned shapes"
CircleArc, ElipseArc, HyperbolaArc, ParabolaArc, LineSegments.


*/

enum ObjectType { LINE = 0 }; // Line for now, but will include Cicle, LineSegment, 
struct Object {
    int type;
    double p[3]; // LINE: a, b, c of a x + b y + c = 0, (a,b) kept unit
};

#ifdef _OPENMP
#pragma omp begin declare target
#endif

// ---------------------------------------------------------- 1. root finding

// smaller of two candidate times that is still ahead of us
inline double firstRoot(double t1, double t2) {
    double t = MAGIC_NO_COLLISION;
    if (t1 > deadTime && t1 < t) t = t1;
    if (t2 > deadTime && t2 < t) t = t2;
    return t;
}

// smallest root > deadTime of A t^2 + B t + C, MAGIC_NO_COLLISION if none
inline double firstQuadRoot(double A, double B, double C) {
    if (A == 0.0) { // no t^2 term: the equation is linear
        if (B == 0.0) return MAGIC_NO_COLLISION;
        return firstRoot(-C / B, MAGIC_NO_COLLISION);
    }

    double disc = B * B - 4.0 * A * C;
    if (disc < 0.0) return MAGIC_NO_COLLISION;

    // stable form: compute q with the sign of B, so B + sq never cancels;
    // one root is q/A, the other C/q
    double sq = sqrt(disc);
    if (B < 0.0) {
        sq = -sq;
    }
    double q = -0.5 * (B + sq);
    if (q == 0.0) { // only when C is exactly 0: the roots are 0 and -B/A
        return firstRoot(0.0, -B / A);
    }
    return firstRoot(q / A, C / q);
}

// ------------------------------------------------------------------ 2. lines

// substitute the parabola into a x + b y + c: A t^2 + B t + C.
// a vertical line has b = 0, killing the t^2 term (linear branch)
inline double lineHitTime(Object o, state s) {
    double a = o.p[0], b = o.p[1], c = o.p[2];
    return firstQuadRoot(-0.5 * g * b,
                         a * s.vx + b * s.vy,
                         a * s.x + b * s.y + c);
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

// --------------------------------------------------------------- 3. dispatch

// time at which the trajectory meets the object, MAGIC_NO_COLLISION if never
inline double hitTime(Object o, state s) {
    switch (o.type) {
    case LINE: return lineHitTime(o, s);
    }
    return MAGIC_NO_COLLISION;
}

// unit normal at the contact point
inline vector2D normalAt(Object o, state s) {
    switch (o.type) {
    case LINE: return lineNormal(o, s);
    }
    return {0.0, 0.0};
}

// project the leftover rounding back onto the surface
inline state snapTo(Object o, state s) {
    switch (o.type) {
    case LINE: return lineSnap(o, s);
    }
    return s;
}

#ifdef _OPENMP
#pragma omp end declare target
#endif

// ----------------------------------------------------------------- 4. makers

// scale the coefficients so that (a, b) is the unit normal
inline Object makeLine(double a, double b, double c) {
    double len = sqrt(a * a + b * b);
    return {LINE, {a / len, b / len, c / len}};
}
