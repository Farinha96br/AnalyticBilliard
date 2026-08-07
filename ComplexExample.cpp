// particles bouncing inside a box, solved analytically collision to collision.
// build and run all three variants (serial, CPU OpenMP, GPU): ./compileAndRun.sh
#include "physics.h" // pulls in geometry.h and types.h
#include <cstdio>
#include <cstdlib>

int main() {

    int Niter = 1e6; // events per particle
    int Npart = 5;

    // the portal pair, named so each can point at the other
    Object portalA = makeLineSegment(-18.5, 2.0, -18.5, 6.0);
    Object portalB = makeLineSegment(-3.0, 2.0, -3.0, 6.0);

    const int nObj = 9;
    Object objs[nObj] = {
        makeLine(0.0, 1.0, 0.0),   // floor    y = 0
        makeLine(0.0, 1.0, -10.0), // ceiling  y = 10
        makeLine(1.0, 0.0, 20.0),  // wall     x = -20
        makeLine(1.0, 0.0, -20.0), // wall     x = 20
        makeLineSegment(0.0, 0.0, 10.0, 5.0),
        makeElipse(-11.0, 5.0, 5.0, 2.0, 0.5), // rotated obstacle, hit from outside
        // an open cup off to the right, clear of both the segment and the ellipse.
        // the gap is the interesting part: particles have to fall in and out of it
        makeElipseArc(15.0, 5.0, 4.0, 2.5, -0.3, 3.4, 6.0),

        // a one-way pair of portals, each naming the other. both span the same
        // heights on purpose: a portal that moved a particle up or down would
        // change its potential energy, and the energy check would stop being a
        // test and start being a known failure
        asPortal(portalA, portalB, 1.0),
        asPortal(portalB, portalA, 1.0)
    };

    // fanned up the x = 0 line, each with its own random velocity. srand is left
    // uncalled so the default seed makes every run repeat exactly
    state *states = new state[Npart];
    for (int i = 0; i < Npart; ++i) {
        states[i] = {0.0, double(i)*1.0,
                     (rand() % 100) / 10.0 - 5.0, (rand() % 100) / 10.0 - 5.0};
    }

    // each particle fills its own slice; nothing prints during the run, since
    // device code cannot stream to stdout. a portal event costs two rows, so the
    // slice is sized for the worst case of every event being one
    int rowsPer = 2 * Niter + 1;
    Row *rows = new Row[Npart * rowsPer];
    int *counts = new int[Npart];

    // one independent particle per thread
#ifdef _OPENMP
#pragma omp target teams distribute parallel for \
    map(to : states[0 : Npart], objs[0 : nObj])  \
    map(from : rows[0 : Npart * rowsPer], counts[0 : Npart])
#endif
    for (int i = 0; i < Npart; ++i) {
        counts[i] = getCollisionStates(states[i], objs, nObj, Niter, &rows[i * rowsPer], rowsPer);
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
        // a portal's surface is described above like any other; this second
        // record carries where it leads, so the plot can draw the link
        if (objs[k].response == PORTAL) {
            printf("# portal %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                   objs[k].q[0], objs[k].q[1], objs[k].q[2], objs[k].q[3],
                   objs[k].q[4], objs[k].q[5], objs[k].q[6], objs[k].q[7],
                   objs[k].q[8], objs[k].q[9]);
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
