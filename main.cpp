#include <iostream>
#include "gay_zone/cuda_include/cuda_metrics.cuh"  // твой заголовок с функциями

int main() {
    float roots[4];
    int n = solveQuartic(1.0f, 0.0f, -5.0f, 0.0f, 4.0f, roots); // x^4 - 5x^2 + 4 = 0

    std::cout << "Найдено корней: " << n << "\n";
    for (int i = 0; i < n; ++i)
        std::cout << "x" << i + 1 << " = " << roots[i] << "\n";

    return 0;
}
