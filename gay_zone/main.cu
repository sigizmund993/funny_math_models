#include "cuda_include/const.cuh"
#include "cuda_include/cuda_auxilary.cuh"
#include "cuda_include/cuda_metrics.cuh"
#include "cuda_include/cuda_solver.cuh"
#include <stdio.h>
int main()
{
    float roots[4];
    int n = solveCubic(1,-4,5,-2,roots);
    printf("n=%d roots: ", n);
    for (int i = 0; i < n; ++i)
        printf("%f ", roots[i]);
    printf("\n");
    return 0;
}
