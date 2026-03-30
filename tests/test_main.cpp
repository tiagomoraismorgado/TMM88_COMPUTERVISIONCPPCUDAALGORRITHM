#include <cassert>
#include <iostream>

int main() {
    // Simple sanity check: basic arithmetic.
    int a = 2;
    int b = 3;
    int c = a + b;
    assert(c == 5);

    std::cout << "Unit test scaffold runs successfully." << std::endl;
    return 0;
}
