#pragma once

// plain aggregates only: they copy to the GPU bitwise, so no constructors,
// no virtuals, no pointers inside

struct vector2D {
    double x, y;
};

struct Roots {
    // a quadratic fills two and a quartic four, so the count travels with the
    // values instead of a sentinel per slot
    double t[4];
    int n;
};

struct state {
    double x;
    double y;
    double vx;
    double vy;
};

struct Row {
    // one line of the output table (the particle id is the buffer position)
    double t, x, y, vx, vy;
};

// constexpr so the values reach the device: a mutable global would live in host
// memory only. gravity is overridable at build time (-DGRAVITY=0.0) because a
// billiard without it is integrable, and those conserved quantities are the
// sharpest test that a new shape reflects correctly
#ifndef GRAVITY
#define GRAVITY 1.0
#endif
constexpr double g = GRAVITY;
constexpr double e = 1.0;                             // restitution: 1.0 is elastic
constexpr double deadTime = 1e-9;                         // roots below this are the bounce we just resolved
constexpr double MAGIC_NO_COLLISION = 987654321000.0; // "never hits" sentinel time
