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

// g and deadTime are mutable so the shared library can set them without a
// rebuild; inline keeps that ODR-safe across translation units, and the
// declare target puts one copy on the device for "target update to" to refresh
#ifdef _OPENMP
#pragma omp declare target
#endif
inline double g = GRAVITY;
inline double deadTime = 1e-9;                        // below this is the bounce just resolved
#ifdef _OPENMP
#pragma omp end declare target
#endif

constexpr double e = 1.0;                             // restitution
constexpr double MAGIC_NO_COLLISION = 987654321000.0;
