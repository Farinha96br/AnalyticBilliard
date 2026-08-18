# Notes

- When performing the escape basins computation, shuffle the initial conditions to destroy spacial correlation, and/or oversubscribe openMP. Do not do it on the GPU.
  - This could be a cool benchmark
- No sine/cosine anywhere to avoid issues with GPU compilers and differnt precision.