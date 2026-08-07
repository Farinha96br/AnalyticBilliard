#pragma once

// plain aggregates only: they copy to the GPU bitwise

struct vector2D {
    double x, y;
};

struct Roots {
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
    double t, x, y, vx, vy; // the particle id is the buffer position
};

// constexpr so the values reach the device. -DGRAVITY=0.0 leaves an integrable
// billiard, whose conserved quantities are the sharpest test of a new shape
#ifndef GRAVITY
#define GRAVITY 1.0
#endif
constexpr double g = GRAVITY;
constexpr double e = 1.0;                             // restitution
constexpr double deadTime = 1e-9;                     // below this is the bounce just resolved
constexpr double MAGIC_NO_COLLISION = 987654321000.0;
