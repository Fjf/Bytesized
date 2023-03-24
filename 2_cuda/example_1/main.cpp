#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include "kernels.cuh"

void print_matrix(std::vector<double> a, struct dims dims) {
    for (size_t i = 0; i < a.size(); i++) {
        if (i % dims.width == 0) std::cout << std::endl;
        printf("%.2f ", a[i]);
    }
    printf("\n");
}

std::vector<double> dissipation(std::vector<double> &sources, struct dims dims, size_t max_iters, double err) {
    auto *output = new std::vector<double>(sources.size());
    auto *buffer = new std::vector<double>(sources.begin(), sources.end());

    std::cout << "Processing dissipation." << std::endl;
    size_t iter;
    for (iter = 0; iter < max_iters; iter++) {
        for (size_t y = 0; y < dims.width; y++) {
            for (size_t x = 0; x < dims.width; x++) {
                /*
                 * Take the four direct neighbours and set current value to sum of
                 *
                 *        0.125
                 *  0.125 0.500 0.125
                 *        0.125
                 *
                 * Edges wrap around.
                 *
                 * The sources are an infinite (fixed) energy point, reset them to their initial values after.
                 */

                if (sources[y * dims.width + x] != 0) {
                    (*output)[y * dims.width + x] = sources[y * dims.width + x];
                } else {
                    uint ny = (y - 1 + dims.height) % dims.height;
                    uint py = (y + 1 + dims.height) % dims.height;
                    uint nx = (x - 1 + dims.width) % dims.width;
                    uint px = (x + 1 + dims.width) % dims.width;
                    (*output)[y * dims.width + x] = (
                            (*buffer)[ny * dims.width + x] * 0.125 +
                            (*buffer)[py * dims.width + x] * 0.125 +
                            (*buffer)[y * dims.width + nx] * 0.125 +
                            (*buffer)[y * dims.width + px] * 0.125 +
                            (*buffer)[y * dims.width + x] * 0.5
                    );
                }
            }
        }
        std::vector<double> *temp = output;
        output = buffer;
        buffer = temp;

        // Check if we converged
        if (equal(*output, *buffer, err)) {
            break;
        }
    }
    std::cout << "Stopped on iteration " << iter << std::endl;
    // The buffer will always contain the last output
    return *buffer;
}

struct dims load_dims(const char *filename) {
    std::cout << "Loading file " << filename << std::endl;
    std::ifstream file(filename);

    std::string line;
    // Get header
    std::getline(file, line);
    size_t width = stoi(line.substr(0, 10));
    size_t height = stoi(line.substr(10, 10));
    struct dims dims = {width, height};
    return dims;
}

std::vector<double> load_data(const char *filename) {
    std::ifstream file;
    file.open(filename, std::ifstream::in);

    std::string line;
    // Get header
    std::getline(file, line);
    size_t width = stoi(line.substr(0, 10));
    size_t height = stoi(line.substr(10, 10));

    // Initialize vector for storage
    std::vector<double> sources(width * height, 0);

    while (std::getline(file, line)) {
        int x = stoi(line.substr(0, 10));
        int y = stoi(line.substr(10, 10));
        double intensity = stod(line.substr(20, 10));

        sources[y * width + x] = intensity;
    }
    return sources;
}

int main(int argc, char **argv) {
    const char *filename = "../example_1/data/heat_points_4";
    bool do_cpu = true;
    if (argc > 1) {
        filename = argv[1];
    }
    if (argc > 2) {
        do_cpu = false;
    }

    /*
     * Initialize data
     */
    struct dims dims = load_dims(filename);
    std::vector<double> sources = load_data(filename);

    /*
     * Constants for computation
     */
    const size_t max_iters = 100000;
    const double err = 1e-5;

    /*
     * Computing reference
     */
    std::vector<double> reference;
    using nano = std::chrono::nanoseconds;

    if (do_cpu) {
        std::cout << "Computing reference on CPU." << std::endl;

        auto ref_start = std::chrono::high_resolution_clock::now();
        reference = dissipation(sources, dims, max_iters, err);
        auto ref_end = std::chrono::high_resolution_clock::now();

        std::cout << "Reference took " << double(std::chrono::duration_cast<nano>(ref_end - ref_start).count()) / 1.e6
                  << " ms\n";
    }

    /*
     * Run kernel
     */
    auto kern_start = std::chrono::high_resolution_clock::now();
    std::vector<double> result = run_kernel(sources, dims, max_iters, err);
    auto kern_end = std::chrono::high_resolution_clock::now();

//    print_matrix(sources, dims);
//    print_matrix(result, dims);
//    print_matrix(reference, dims);

    std::cout << "Kernel took " << double(std::chrono::duration_cast<nano>(kern_end - kern_start).count()) / 1.e6
              << " ms\n";
    /*
     * Validate and print result.
     */
    if (do_cpu) {
        std::cout << "Result: " << std::boolalpha << equal(reference, result, err) << std::endl;
    }
    return 0;
}
