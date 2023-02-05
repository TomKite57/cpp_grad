
#include<iostream>

#include "src/value.hpp"


int main()
{
    using TYPE = double;

    auto a = Value<TYPE>(2.0);
    auto b = Value<TYPE>(-3.0);
    auto c = Value<TYPE>(10.0);
    auto e = a * b;
    auto d = e + c;
    auto f = Value<TYPE>(-2.0);
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
