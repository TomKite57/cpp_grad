
#include<iostream>

#include "src/value.hpp"
//#include "src/module.hpp"


int main()
{
    auto a = Value<double>(2.0);
    auto b = Value<double>(-3.0);
    auto c = Value<double>(10.0);
    auto e = a * b;
    auto d = e + c;
    auto f = Value<double>(-2.0);
    auto L = d * f;

    auto T = Value<double>(6.0);
    T = T + L;
    T = T + f;
    T = T + d;

    L = T;

    auto order = T.build_topo();
    for (auto n=order.begin(); n!=order.end(); ++n)
        std::cout << **n << " | " << *n << "\n";
    std::cout << "\n";

    L.backward();

    //std::cout << a << " | " << a.get_ptr() << "\n"
    //          << b << " | " << b.get_ptr() << "\n"
    //          << c << " | " << c.get_ptr() << "\n"
    //          << d << " | " << d.get_ptr() << "\n"
    //          << e << " | " << e.get_ptr() << "\n"
    //          << f << " | " << f.get_ptr() << "\n"
    //          << L << " | " << L.get_ptr() << "\n";
    //std::cout << "\n";


    for (auto n=order.begin(); n!=order.end(); ++n)
        std::cout << **n << " | " << *n << "\n";
    std::cout << "\n";

    
    //for (auto& n : build_topo(&L))
    //    std::cout << *n << " | " << n << "\n";

   /*
    MLP<TYPE> model({2, 2, 2, 2});

    std::vector<TYPE> input = {1, 1};
    std::vector<TYPE> target = {1, 0};
    auto loss = model.loss(input, target);
    */

    return 0;
}
