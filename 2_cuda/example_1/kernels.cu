#include <vector>
#include <iostream>
#include "kernels.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

//#define HALO
//#define SHARED
#ifdef HALO
#define HALO_OFFSET 1
#else
#define HALO_OFFSET 0
#endif

#define BLOCK_SIZE 32


__global__ void
base_kernel(const double *sources, double *output, const double *buffer, struct dims dims, double err, bool *is_equal) {
    long x = blockIdx.x * blockDim.x + threadIdx.x;
    long y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dims.width || y >= dims.height) return;

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
        output[y * dims.width + x] = sources[y * dims.width + x];
    } else {
        uint ny = (y - 1 + dims.height) % dims.height;
        uint py = (y + 1 + dims.height) % dims.height;
        uint nx = (x - 1 + dims.width) % dims.width;
        uint px = (x + 1 + dims.width) % dims.width;

        output[y * dims.width + x] = (
                buffer[ny * dims.width + x] * 0.125 +
                buffer[py * dims.width + x] * 0.125 +
                buffer[y * dims.width + nx] * 0.125 +
                buffer[y * dims.width + px] * 0.125 +
                buffer[y * dims.width + x] * 0.5
        );
    }

    if (std::fabs(buffer[y * dims.width + x] - output[y * dims.width + x]) > err) {
        *is_equal = false;
    }
}


__global__ void kernel_shared_mem(const double *sources, double *output, const double *buffer, struct dims dims,
                                  double err, bool *is_equal) {
    long globalX = blockIdx.x * blockDim.x + threadIdx.x;
    long globalY = blockIdx.y * blockDim.y + threadIdx.y;

    if (globalX >= dims.width || globalY >= dims.height) return;

    // Compute offset due to halo cell region


    // Load into shared memory
    long bufferHeight = min(int(dims.height), BLOCK_SIZE);
    long bufferWidth = min(int(dims.width), BLOCK_SIZE);
    __shared__ double localBuffer[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    localBuffer[threadIdx.y + 1][threadIdx.x + 1] = buffer[globalY * dims.width + globalX];
    // Load edges
    auto* localBufferPtr = reinterpret_cast<double *>(&localBuffer);

    uint ny = (blockIdx.y * blockDim.y - 1 + dims.height) % dims.height;
    uint py = ((blockIdx.y + 1) * blockDim.y + dims.height) % dims.height;
    uint nx = (blockIdx.x * blockDim.x - 1 + dims.width) % dims.width;
    uint px = ((blockIdx.x + 1) * blockDim.x + dims.width) % dims.width;

    /*
     *
     */
    if (threadIdx.y == 0) {
        localBuffer[0][threadIdx.x + 1] = buffer[ny * dims.width + globalX];
//        printf("%d reads from %d %d\n", threadIdx.x, ny, int(globalX));
    } else if (threadIdx.y == 1) {
        localBuffer[bufferHeight + 2 - 1][threadIdx.x + 1] = buffer[py * dims.width + globalX];
    } else if (threadIdx.y == 2) {
        localBuffer[threadIdx.x + 1][0] = buffer[(blockIdx.y * blockDim.y + threadIdx.x) * dims.width + nx];
    } else if (threadIdx.y == 3) {
        localBuffer[threadIdx.x + 1][bufferWidth + 2 - 1] = buffer[(blockIdx.y * blockDim.y + threadIdx.x) * dims.width +
                                                                  px];
    }


    // Wait for all threads to finish loading data
    __syncthreads();

    long x = threadIdx.x + 1;
    long y = threadIdx.y + 1;

    if (sources[globalY * dims.width + globalX] != 0) {
        output[globalY * dims.width + globalX] = sources[globalY * dims.width + globalX];
    } else {
        output[globalY * dims.width + globalX] = (
                localBuffer[y - 1][x] * 0.125 +
                localBuffer[y + 1][x] * 0.125 +
                localBuffer[y][x - 1] * 0.125 +
                localBuffer[y][x + 1] * 0.125 +
                localBuffer[y][x] * 0.5
        );
    }

    if (std::fabs(buffer[globalY * dims.width + globalX] - output[globalY * dims.width + globalX]) > err) {
        *is_equal = false;
    }
}

__global__ void
halo_kernel(const double *sources, double *h_output, const double *h_buffer, struct dims dims, double err,
            bool *is_equal) {
    long x = blockIdx.x * blockDim.x + threadIdx.x;
    long y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dims.width || y >= dims.height) return;



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


    long data_width = dims.width + HALO_OFFSET * 2;

    // Compute offset due to halo cell region
    const double *buffer = h_buffer + (HALO_OFFSET * data_width) + HALO_OFFSET;
    double *output = h_output + (HALO_OFFSET * data_width) + HALO_OFFSET;

    long ny = (y - 1);
    long py = (y + 1);

    long nx = (x - 1);
    long px = (x + 1);
    if (sources[y * dims.width + x] != 0) {
        output[y * data_width + x] = sources[y * dims.width + x];
    } else {
        output[y * data_width + x] = (
                buffer[ny * data_width + x] * 0.125 +
                buffer[py * data_width + x] * 0.125 +
                buffer[y * data_width + nx] * 0.125 +
                buffer[y * data_width + px] * 0.125 +
                buffer[y * data_width + x] * 0.5
        );
    }

    if (std::fabs(buffer[y * data_width + x] - output[y * data_width + x]) > err) {
        *is_equal = false;
    }
}

__global__ void set_halo(double *h_buffer, struct dims dims) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    long data_width = dims.width + 2;

    // Offset into real data instead of halo cells
    double *buffer = h_buffer + (HALO_OFFSET * data_width) + HALO_OFFSET;
    if (i < dims.width) {
        // Set buffer for ys
        buffer[(-1) * data_width + i] = buffer[(dims.height - 1) * data_width + i];
        buffer[(dims.height) * data_width + i] = buffer[(0) * data_width + i];
    }

    if (i < dims.height) {
        // Set buffer for xs
        buffer[i * data_width + (-1)] = buffer[i * data_width + (dims.width - 1)];
        buffer[i * data_width + (dims.width)] = buffer[i * data_width + (0)];
    }
}

__global__ void copy_to(const double *sources, double *output, double *buffer, struct dims dims) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dims.width || y >= dims.height) return;

    uint data_width = dims.width + (HALO_OFFSET * 2);
    uint cy = y + HALO_OFFSET;
    uint cx = x + HALO_OFFSET;
    buffer[cy * data_width + cx] = sources[y * dims.width + x];
    output[cy * data_width + cx] = sources[y * dims.width + x];
}

__global__ void copy_back(double *sources, const double *buffer, struct dims dims) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dims.width || y >= dims.height) return;

    uint data_width = dims.width + (HALO_OFFSET * 2);
    uint cy = y + HALO_OFFSET;
    uint cx = x + HALO_OFFSET;
    sources[y * dims.width + x] = buffer[cy * data_width + cx];
}

bool equal(std::vector<double> &a, std::vector<double> &b, double err, bool do_print) {
    for (size_t i = 0; i < a.size(); i++) {
        if (std::fabs(a[i] - b[i]) > err) {
            if (do_print) {
                printf("%zu: %.2f != %.2f\n", i, a[i], b[i]);
            }
            return false;
        }
    }
    return true;
}

std::vector<double>
run_kernel(const std::vector<double> &sources, const struct dims dims, const size_t max_iters, const double err) {
    // Initialize vector for our output array.
    std::vector<double> output(sources.size());
    std::vector<double> buffer(sources.size());
    /*
     * Create device pointer and calculate buffer size.
     */
    double *dev_sources;
    double *dev_buffer;
    double *dev_output;
    bool *dev_is_equal;
    size_t input_size = sources.size() * sizeof(double);
    size_t buffer_size = (dims.width + HALO_OFFSET * 2) * (dims.height + HALO_OFFSET * 2) * sizeof(double);

    /*
     * Allocate and copy memory to device.
     */
    gpuErrchk(cudaMalloc((void **) &dev_sources, input_size));
    gpuErrchk(cudaMalloc((void **) &dev_output, buffer_size));
    gpuErrchk(cudaMalloc((void **) &dev_buffer, buffer_size));
    gpuErrchk(cudaMalloc((void **) &dev_is_equal, sizeof(bool)));
    gpuErrchk(cudaMemcpy((void *) dev_sources, (void *) sources.data(), input_size, cudaMemcpyHostToDevice));

    /*
     * Calculate block size for our data.
     */
    uint block_size = BLOCK_SIZE;
    dim3 block = {block_size, block_size};
    dim3 grid = {
            uint((dims.width + block.x) / block.x),
            uint((dims.height + block.y) / block.y)
    };

    /*
     * Calculate block size for the halo cell copying
     */
    dim3 hblock = {block_size * block_size};
    dim3 hgrid = {
            uint((dims.width + block.x) / block.x),
    };


    // Copy over all data into the correct buffer locations
    copy_to<<<grid, block>>>(dev_sources, dev_output, dev_buffer, dims);
    gpuErrchk(cudaDeviceSynchronize());

    /*
     * Iterate computation
     */
    size_t iter;
    bool is_equal;
    std::cout << "[GPU] Processing dissipation." << std::endl;
    for (iter = 0; iter < max_iters; iter++) {
        /*
         * Launch our kernel
         */

        gpuErrchk(cudaMemset((void *) dev_is_equal, 1, 1));

#ifdef HALO
        set_halo<<<hgrid, hblock>>>(dev_buffer, dims);
        gpuErrchk(cudaDeviceSynchronize());

        halo_kernel<<<grid, block>>>(dev_sources, dev_output, dev_buffer, dims, err, dev_is_equal);
        gpuErrchk(cudaDeviceSynchronize());
#else
#ifdef SHARED
        kernel_shared_mem<<<grid, block>>>(dev_sources, dev_output, dev_buffer, dims, err, dev_is_equal);
        gpuErrchk(cudaDeviceSynchronize());
#else
        base_kernel<<<grid, block>>>(dev_sources, dev_output, dev_buffer, dims, err, dev_is_equal);
        gpuErrchk(cudaDeviceSynchronize());
#endif
#endif
        cudaMemcpy((void *) &is_equal, (void *) dev_is_equal, sizeof(bool), cudaMemcpyDeviceToHost);

        /*
         * Swap buffers
         */
        double *temp = dev_output;
        dev_output = dev_buffer;
        dev_buffer = temp;

        if (is_equal)
            break;
    }
    std::cout << "[GPU] Stopped on iteration " << iter << std::endl;

    // Copy the data with halo cells back to an array with easier shape
    copy_back<<<grid, block>>>(dev_sources, dev_buffer, dims);
    gpuErrchk(cudaDeviceSynchronize());


    /*
     * Copy our data back to host.
     */
    cudaMemcpy((void *) output.data(), (void *) dev_sources, input_size, cudaMemcpyDeviceToHost);
    return output;
}
