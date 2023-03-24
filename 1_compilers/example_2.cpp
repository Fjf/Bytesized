#include <iostream>
#include <chrono>


int main() {
    int a = 32;
    int b = 46;

    u_long final_result = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1e9; i++) {
        int result = a * b + 4;
        final_result += result;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Result: " << final_result << " in " << duration << "ms" << std::endl;

    return 0;
}
