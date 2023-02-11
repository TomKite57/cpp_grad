
#include<iostream>
#include<cstdlib>

#include "src/value.hpp"
#include "src/module.hpp"
#include "src/utils.hpp"

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
    set_seed();
    
    sanity_check();
    
    //std::vector<double> input = {4.7, 5.0};
    std::vector<Value<double>> input{Value<double>(4.7), Value<double>(5.0)};

    //Neuron<double> neuron(2);
    //auto val = neuron(input);
    //std::cout << val << std::endl;

    //Layer<double> layer(2, 2);
    //auto val = layer(input);
    //std::cout << val << std::endl;

    MLP<double> model({2, 2, 2, 2});
    std::vector<double> target = {1, 0};

    for (size_t i=0; i<100; ++i)
    {
        auto loss = model.loss(input, target);
        loss.backward();
        model.descend_grad();
        model.zero_grad();
        std::cout << loss << std::endl;
    }

    return 0;
}
