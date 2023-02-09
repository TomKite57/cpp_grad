
#include<iostream>

#include "src/value.hpp"
//#include "src/module.hpp"


int main()
{
    using TYPE = double;

    auto a = Value<TYPE>(2.0);
    auto b = Value<TYPE>(-3.0);
    auto c = Value<TYPE>(10.0);
    auto e = a * b;
    auto d = (e + c) * a;
    auto f = Value<TYPE>(-2.0);
    auto L = d * f;

    //L = L * L;
    
    L.backward();

    for (auto& par : L.get_parent_ptrs())
        std::cout << *par << " | " << par << std::endl;

    std::cout << "\n";

    auto order = build_topo(&L);
    for (auto n=order.rbegin(); n!=order.rend(); ++n)
    {
        std::cout << **n << " | " << *n << "\n";
    }

    std::cout << "\n";
    
    std::cout << a << " | " << &a << "\n"
              << b << " | " << &b << "\n"
              << c << " | " << &c << "\n"
              << d << " | " << &d << "\n"
              << e << " | " << &e << "\n"
              << f << " | " << &f << "\n"
              << L << " | " << &L << "\n\n";
    
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
