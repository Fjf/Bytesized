#include <iostream>
#include "cuda.h"


struct dims {
    size_t width;
    size_t height;
};

bool equal(std::vector<double> &a, std::vector<double> &b, double err, bool do_print=false);

std::vector<double> run_kernel(const std::vector<double> &sources, struct dims dims, size_t max_iters, double err);