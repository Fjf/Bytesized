#include <iostream>
#include <chrono>


int foo(int a) {
    return a % 32 == 0 ? 1 : 78;
}

int foo2(int a) {
    a *= 2;
    int result1 = 1;
    int result2 = 78;
    a /= 2;
    if (a % 32 == 0) {
        return result1;
    } else {
        return result2;
    }
}


int main() {
    // Define constant
    int a = 32;
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

    // Profile second function (foo2)
    final_result = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1e9; i++) {
        int result = foo2(a);
        a += increment;
        final_result += result;
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Result: " << final_result << " in " << duration << "ms" << std::endl;


    return 0;
}
