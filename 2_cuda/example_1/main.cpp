#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include "kernels.cuh"


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
                    (*output)[y * dims.width + x] = (
                            (*buffer)[((y + 1) % dims.height) * dims.width + ((x + 0) % dims.width)] * 0.125 +
                            (*buffer)[((y - 1 + dims.width) % dims.height) * dims.width + ((x + 0) % dims.width)] *
                            0.125 +
                            (*buffer)[((y + 0) % dims.height) * dims.width + ((x + 1) % dims.width)] * 0.125 +
                            (*buffer)[((y + 0) % dims.height) * dims.width + ((x - 1 + dims.width) % dims.width)] *
                            0.125 +
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
    if (argc > 1) {
        filename = argv[1];
    }
    /*
     * Initialize data
     */
    struct dims dims = load_dims(filename);
    std::vector<double> sources = load_data(filename);

    /*
     * Constants for computation
     */
    const size_t max_iters = 10000;
    const double err = 1e-4;

    /*
     * Computing reference
     */

    using nano = std::chrono::nanoseconds;
    std::cout << "Computing reference on CPU." << std::endl;

    auto ref_start = std::chrono::high_resolution_clock::now();
    std::vector<double> reference = dissipation(sources, dims, max_iters, err);
    auto ref_end = std::chrono::high_resolution_clock::now();

    std::cout << "Reference took " << double(std::chrono::duration_cast<nano>(ref_end - ref_start).count()) / 1.e6
              << " ms\n";

    /*
     * Run kernel
     */
    auto kern_start = std::chrono::high_resolution_clock::now();
    std::vector<double> result = run_kernel(sources, dims, max_iters, err);
    auto kern_end = std::chrono::high_resolution_clock::now();

    std::cout << "Kernel took " << double(std::chrono::duration_cast<nano>(kern_end - kern_start).count()) / 1.e6
              << " ms\n";
    /*
     * Validate and print result.
     */
    std::cout << "Result: " << std::boolalpha << equal(reference, result, 1e4) << std::endl;
    return 0;
}
