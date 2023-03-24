#include <iostream>
#include <chrono>


int foo(u_long a) {
    return a % 32 == 0 ? 1 : 78;
}

int main(int argc, char** argv) {
    u_long a = atoi(argv[1]);
    // Define constant
    int increment = 32;

    // Profile first function (foo)
    u_long final_result = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1e9; i++) {
        int result = foo(a);
        a += increment;
        final_result += result;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Result: " << final_result << " in " << duration << "ms" << std::endl;

    return 0;
}
