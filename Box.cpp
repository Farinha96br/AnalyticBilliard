// particles bouncing inside a box, solved analytically collision to collision.
// build and run all three variants (serial, CPU OpenMP, GPU): ./compileAndRun.sh
#include "physics.h" // pulls in geometry.h and types.h
#include <cstdio>
#include <cstdlib>

#define MAGIC_CHECK 123456789

int main(int argc, char **argv) {

    int Niter = 1e4; // number of bounces per particle
    int Npart = 5;  // number of particles

    // create a box of four planes:
    const int nObj = 7;
    Object objs[nObj] = {
        makeLine(0.0, 1.0, 0.0),   // floor    y = 0
        makeLine(0.0, 1.0, -10.0), // ceiling  y = 20
        makeLine(1.0, 0.0, 20.0),  // wall     x = -20
        makeLine(1.0, 0.0, -20.0), // wall     x = 20
        makeLineSegment(0.0, 0.0, 10.0, 5.0), // Line segment from (0,0) to (10,5)
        makeElipse(-11.0, 5.0, 5.0, 2.0, 0.5), // rotated obstacle, hit from outside
        // an open cup off to the right, clear of both the segment and the ellipse.
        // the gap is the interesting part: particles have to fall in and out of it
        makeElipseArc(15.0, 5.0, 4.0, 2.5, -0.3, 3.4, 6.0)
    };

    //
    // initialize the states with random velicities from a single point:

    state *states = new state[Npart]; // allocate memory for the states
    for (int i = 0; i < Npart; ++i) {
        states[i] = {0.0, double(i)*1.0, (rand() % 100) / 10.0 - 5.0, (rand() % 100) / 10.0 - 5.0}; // random velocities
    }

    // every particle fills its own slice of this buffer; nothing prints
    // during the run (device code cannot stream to stdout anyway)
    int rowsPer = Niter + 1; // initial condition + one row per bounce
    Row *rows = new Row[Npart * rowsPer];
    int *counts = new int[Npart];

    // the simulation: one independent particle per thread
#ifdef _OPENMP
#pragma omp target teams distribute parallel for \
    map(to : states[0 : Npart], objs[0 : nObj])  \
    map(from : rows[0 : Npart * rowsPer], counts[0 : Npart])
#endif
    for (int i = 0; i < Npart; ++i) {
        counts[i] = getCollisionStates(states[i], objs, nObj, Niter, &rows[i * rowsPer]);
    }

    // write everything at once: the scene, then one row per collision.
    // each shape emits its own tag and its own parameters: p[] means something
    // different per type, so a single fixed format would misread every object
    for (int k = 0; k < nObj; ++k) {
        switch (objs[k].type) {
        case LINE:
            printf("# line %.17g %.17g %.17g\n",
                   objs[k].p[0], objs[k].p[1], objs[k].p[2]);
            break;
        case LINE_SEGMENT:
            printf("# segment %.17g %.17g %.17g %.17g\n",
                   objs[k].p[0], objs[k].p[1], objs[k].p[2], objs[k].p[3]);
            break;
        case ELIPSE: // only the five real parameters: cos/sin are derived from theta
            printf("# elipse %.17g %.17g %.17g %.17g %.17g\n",
                   objs[k].p[0], objs[k].p[1], objs[k].p[2], objs[k].p[3], objs[k].p[4]);
            break;
        case ELIPSE_ARC: // the ellipse's five, then the two angular bounds
            printf("# elipsearc %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                   objs[k].p[0], objs[k].p[1], objs[k].p[2], objs[k].p[3], objs[k].p[4],
                   objs[k].p[7], objs[k].p[8]);
            break;
        }
    }
    printf("ID t x y vx vy\n");
    for (int i = 0; i < Npart; ++i) {
        for (int r = 0; r < counts[i]; ++r) {
            Row w = rows[i * rowsPer + r];
            printf("%d %.17g %.17g %.17g %.17g %.17g\n", i, w.t, w.x, w.y, w.vx, w.vy);
        }
    }

    delete[] states;
    delete[] rows;
    delete[] counts;

    return 0;
}
