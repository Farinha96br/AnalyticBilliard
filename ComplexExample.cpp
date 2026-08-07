// particles bouncing inside a box, solved analytically collision to collision.
// build and run all three variants (serial, CPU OpenMP, GPU): ./compileAndRun.sh
#include "physics.h" // pulls in geometry.h and types.h
#include <cstdio>
#include <cstdlib>

int main() {

    int Niter = 1e6; // events per particle
    int Npart = 5;

    Object portalA = makeLineSegment(-18.5, 2.0, -18.5, 6.0);
    Object portalB = makeLineSegment(-3.0, 2.0, -3.0, 6.0);

    const int nObj = 9;
    Object objs[nObj] = {
        makeLine(0.0, 1.0, 0.0),   // floor    y = 0
        makeLine(0.0, 1.0, -10.0), // ceiling  y = 10
        makeLine(1.0, 0.0, 20.0),  // wall     x = -20
        makeLine(1.0, 0.0, -20.0), // wall     x = 20
        makeLineSegment(0.0, 0.0, 10.0, 5.0),
        makeElipse(-11.0, 5.0, 5.0, 2.0, 0.5),
        makeElipseArc(15.0, 5.0, 4.0, 2.5, -0.3, 3.4, 6.0),

        // both span the same heights on purpose: a portal that moved a particle
        // up or down would change its potential energy, and the energy check
        // would stop being a test and start being a known failure
        asPortal(portalA, portalB, 1.0),
        asPortal(portalB, portalA, 1.0)
    };

    // srand is left uncalled so the default seed makes every run repeat exactly
    state *states = new state[Npart];
    for (int i = 0; i < Npart; ++i) {
        states[i] = {0.0, double(i)*1.0,
                     (rand() % 100) / 10.0 - 5.0, (rand() % 100) / 10.0 - 5.0};
    }

    // nothing prints during the run: device code cannot stream to stdout. a
    // portal event costs two rows, so the slice is sized for the worst case
    int rowsPer = 2 * Niter + 1;
    Row *rows = new Row[Npart * rowsPer];
    int *counts = new int[Npart];

#ifdef _OPENMP
#pragma omp target teams distribute parallel for \
    map(to : states[0 : Npart], objs[0 : nObj])  \
    map(from : rows[0 : Npart * rowsPer], counts[0 : Npart])
#endif
    for (int i = 0; i < Npart; ++i) {
        counts[i] = getCollisionStates(states[i], objs, nObj, Niter, &rows[i * rowsPer], rowsPer);
    }

    // p[] means something different per type, so a single fixed format would
    // misread every object; a portal adds a second record saying where it leads
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
        case ELIPSE:
            printf("# elipse %.17g %.17g %.17g %.17g %.17g\n",
                   objs[k].p[0], objs[k].p[1], objs[k].p[2], objs[k].p[3], objs[k].p[4]);
            break;
        case ELIPSE_ARC:
            printf("# elipsearc %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                   objs[k].p[0], objs[k].p[1], objs[k].p[2], objs[k].p[3], objs[k].p[4],
                   objs[k].p[7], objs[k].p[8]);
            break;
        }
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
