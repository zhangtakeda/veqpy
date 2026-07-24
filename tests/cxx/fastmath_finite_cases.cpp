#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "veq_numeric.h"

int main(int argc, char** argv)
{
    if (argc != 2)
        return 2;

    errno            = 0;
    char* end        = nullptr;
    const auto value = std::strtoull(argv[1], &end, 16);
    if (errno != 0 || end == argv[1] || *end != '\0')
        return 3;

    const auto bits = static_cast<std::uint64_t>(value);
    std::cout << math::is_finite(std::bit_cast<double>(bits)) << '\n';
    return 0;
}
