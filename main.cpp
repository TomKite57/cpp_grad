
#include<iostream>

#include "src/value.hpp"
#include "src/module.hpp"

void sanity_check()
{
    auto a = Value<double>(2.0);
    auto b = Value<double>(-3.0);
    auto c = Value<double>(10.0);
    auto e = a * b;
    auto d = e + c;
    auto f = Value<double>(-2.0);
    auto L = d * f;

    L.backward();

    std::cout << a << " | " << a.get_ptr() << "\n"
              << b << " | " << b.get_ptr() << "\n"
              << c << " | " << c.get_ptr() << "\n"
              << d << " | " << d.get_ptr() << "\n"
              << e << " | " << e.get_ptr() << "\n"
              << f << " | " << f.get_ptr() << "\n"
              << L << " | " << L.get_ptr() << "\n";
    std::cout << "\n";
}

int main()
{
    //sanity_check();

    MLP<double> model({2, 2, 2, 2});

    std::vector<double> input = {1, 1};
    std::vector<double> target = {1, 0};
    auto loss = model.loss(input, target);

    return 0;
}
