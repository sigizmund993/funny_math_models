#include <vector>
#include <cmath>

using namespace std;

void my_swap(double &a, double &b) {
    static double c;
    c = a;
    a = b;
    b = c;
}

int partition(double *arr, int low, int high) {
    static double pivot;
    static int i, j;
    pivot = arr[high];
    i = low - 1;
    for (j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            if (i != j) {
                my_swap(arr[i], arr[j]);
            }
        }
    }
    i++;
    if (i != j) {
        my_swap(arr[i], arr[j]);
    }
    return i;
}

void quick_sort(double *arr, int low, int high) {
    static int pivot;
    if (low < high) {
        pivot = partition(arr, low, high);
        quick_sort(arr, low, pivot - 1);
        quick_sort(arr, pivot + 1, high);
    }
}

int abs_partition(vector<double> &arr, int low, int high) {
    static double pivot;
    static int i, j;
    pivot = fabs(arr[high]);
    i = low - 1;
    for (j = low; j < high; j++) {
        if (fabs(arr[j]) < pivot) {
            i++;
            if (i != j) {
                my_swap(arr[i], arr[j]);
            }
        }
    }
    i++;
    if (i != j) {
        my_swap(arr[i], arr[j]);
    }
    return i;
}

void abs_sort(vector<double> &arr, int low, int high) {
    static int pivot;
    if (low < high) {
        pivot = abs_partition(arr, low, high);
        abs_sort(arr, low, pivot - 1);
        abs_sort(arr, pivot + 1, high);
    }
}