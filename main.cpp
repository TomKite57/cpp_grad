
#include<iostream>

#include "src/value.hpp"


int main()
{
    auto a = Value<double>(2.0);
    auto b = Value<double>(-3.0);
    auto c = Value<double>(10.0);
    auto e = a * b;
    auto d = e + c;
    auto f = Value<double>(-2.0);
    auto L = d * f;

    L.backward();

    std::cout << a << "\n"
              << b << "\n"
              << c << "\n"
              << d << "\n"
              << e << "\n"
              << f << "\n"
              << L << "\n";

    return 0;
}
