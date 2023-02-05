
#include<iostream>

#include "src/value.hpp"


int main()
{
    auto a = Value<double>(2.0);
    auto b = Value<double>(-3.0);
    //auto c = Value<double>(10.0);
    auto c = a / b - 3.0;

    auto d = pow(c, 2.0);
    //auto d = c * 3.0;

    d.backward();

    std::cout << a << "\n"
              << b << "\n"
              << c << "\n"
              << d << "\n";

    return 0;
}
