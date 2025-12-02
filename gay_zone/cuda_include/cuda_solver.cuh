#include "const.cuh"
#include <stdio.h>
__host__ __device__ int solveSquare(float a, float b, float c, float roots[2])
{
    float D = b * b - 4.0f * a * c;
    if (D < 0.0f) return 0;
    if (fabsf(D) < 1e-8f) {
        roots[0] = -b / (2.0f * a);
        return 1;
    }
    float sqrtD = sqrtf(D);
    roots[0] = (-b + sqrtD) / (2.0f * a);
    roots[1] = (-b - sqrtD) / (2.0f * a);
    return 2;
}
__host__ __device__ int solveCubic(float a, float b, float c, float d, float roots[3])//
{
    if (fabsf(a) < 1e-8f) { 
        return solveSquare(b,c,d,roots);
    }
    //x^3+Ax^2+Bx+c = 0
    float A = b/a;
    float B = c/a;
    float C = d/a;
    //x = y+2
    //y^3+py+q = 0
    float p = B-A*A/3;
    float q = C + 2*A*A*A/27-A*B/3;
    //y = t^(1/3)+6/t^(1/3)
    //t^2 + q*t - p^3/27 = 0
    float rootst[2];
    printf("%f,%f,%f\n",1.0,q,-p*p*p/27);
    int n = solveSquare(1,q,-p*p*p/27,rootst);
    printf("%i\n",n);
    for (int i = 0; i < n; ++i)
        printf("%f ", roots[i]);
    printf("\n");
    float x = cbrtf(rootst[0])+cbrtf(rootst[1]) + 2;
    roots[0] = x;
    return 1;
    // float roots2[2];
    // int answ = solveSquare(a,b+x*a,c+x*b+x*x*a,roots2);
    // for (int i = 0; i < answ; ++i)
    //     roots[i+1] = roots2[i];
    // return answ+1;
}


// __host__ __device__ int solveQuartic(float a, float b, float c, float d, float e, float roots[4])
// {
//     if (fabsf(a) < 1e-8f) {
//         return solveCubic(b, c, d, e, roots);
//     }

//     float invA = 1.0f / a;
//     float A = b * invA;
//     float B = c * invA;
//     float C = d * invA;
//     float D = e * invA;

//     float A2 = A * A;
//     float p = -3.0f * A2 / 8.0f + B;
//     float q = A2 * A / 8.0f - (A * B) / 2.0f + C;
//     float r = -3.0f * A2 * A2 / 256.0f + (A2 * B) / 16.0f - (A * C) / 4.0f + D;

//     float rootsC[3];
//     float z, u, v;
//     float yRoots[4];
//     int nRoots = 0;

//     if (fabsf(q) < 1e-8f) {
//         float D0 = p * p - 4.0f * r;
//         if (D0 < 0.0f) return 0;
//         float sqrtD0 = sqrtf(D0);
//         float y1 = (-p + sqrtD0) / 2.0f;
//         float y2 = (-p - sqrtD0) / 2.0f;
//         float vals[4];
//         int count = 0;
//         if (y1 >= 0.0f) {
//             float s = sqrtf(y1);
//             vals[count++] = s;
//             vals[count++] = -s;
//         }
//         if (y2 >= 0.0f) {
//             float s = sqrtf(y2);
//             vals[count++] = s;
//             vals[count++] = -s;
//         }
//         for (int i = 0; i < count; ++i)
//             roots[i] = vals[i] - A / 4.0f;
//         return count;
//     }

//     float cubicA = 1.0f;
//     float cubicB = 0.5f * p;
//     float cubicC = 0.25f * ((p * p) - 4.0f * r);
//     float cubicD = -0.015625f * (q * q); // q^2 / 64

//     int nc = solveCubic(cubicA, cubicB, cubicC, cubicD, rootsC);
//     if (nc < 1) return 0;

//     z = rootsC[0];

//     if (z < 0.0f) z = 0.0f;

//     float u2 = 2.0f * z - p;
//     if (u2 < 0.0f) u2 = 0.0f;
//     u = sqrtf(u2);

// //    float v2 = (q < 0.0f ? -q : q);
//     if (u != 0.0f)
//         v = -q / (2.0f * u);
//     else
//         v = 0.0f;


//     float D1 = u * u - 4.0f * (z - v);
//     float D2 = u * u - 4.0f * (z + v);

//     if (D1 >= 0.0f) {
//         float sqrtD1 = sqrtf(D1);
//         yRoots[nRoots++] = (-u + sqrtD1) / 2.0f;
//         yRoots[nRoots++] = (-u - sqrtD1) / 2.0f;
//     }
//     if (D2 >= 0.0f) {
//         float sqrtD2 = sqrtf(D2);
//         yRoots[nRoots++] = (u + sqrtD2) / 2.0f;
//         yRoots[nRoots++] = (u - sqrtD2) / 2.0f;
//     }

//     for (int i = 0; i < nRoots; ++i)
//         roots[i] = yRoots[i] - A / 4.0f;

//     return nRoots;
// }
