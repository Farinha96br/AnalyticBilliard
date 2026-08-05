#pragma once

// plain aggregates only: they copy to the GPU bitwise, so no constructors,
// no virtuals, no pointers inside

struct vector2D {
    /*
    simple 2d vector to return two values
    */
    double x, y;
};

struct Roots {
    /*
    up to four real roots. a quadratic fills two and a quartic four, so the
    count travels with the values instead of a sentinel per slot
    */
    double t[4];
    int n; // how many entries of t[] are valid
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

// constexpr so the values exist on the device at compile time
// (a mutable global would live in host memory only)
// gravity, pulling towards -y. overridable at build time (-DGRAVITY=0.0) because
// a billiard without gravity is integrable, and those conserved quantities are the
// sharpest test there is that a new shape reflects correctly
#ifndef GRAVITY
#define GRAVITY 1.0
#endif
constexpr double g = GRAVITY;
constexpr double e = 1.0;                             // restitution: 1.0 is elastic
constexpr double deadTime = 1e-9;                         // roots below this are the bounce we just resolved
constexpr double MAGIC_NO_COLLISION = 987654321000.0; // "never hits" sentinel time
