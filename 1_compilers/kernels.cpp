#include <iostream>


int foo(int a, int b) {
    return a * b + 4;
}


int main() {
    int a = 32;
    int b = 46;

    int result = foo(a, b);
    std::cout << "Result: " << result << std::endl;
    return 0;
}
