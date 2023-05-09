
#include<iostream>
#include<cstdlib>
#include<tuple>

#include "src/value.hpp"
#include "src/module.hpp"
#include "src/static_module.hpp"
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

void MLP_test(size_t iter)
{
    MLP<double> model({5, 5, 5, 5});

    std::vector<double> input = {4.7, 5.0, 5.2, 5.4, 5.6};
    std::vector<double> target = {1, 0, 0, 0, 0};

    for (size_t i=0; i<iter; ++i)
    {
        auto loss = model.loss(input, target);
        loss.backward();

        if (i%(iter/100)==0)
        {
            model.descend_grad(0.00001);
            model.zero_grad();
        }

        if (i%(iter/5)==0)
            std::cout << "Output: " << model(input) << "\nLoss: " << loss << "\n";
    }
}

void static_MLP_test(size_t iter)
{
    constexpr size_t size = 5;
    using l = StaticLayer<double, size, size>;
    StaticModule<l, l, l, l> model{};

    std::array<Value<double>, size> input = {4.7, 5.0, 5.2, 5.4, 5.6};
    std::array<Value<double>, size> target = {1, 0, 0, 0, 0};

    for (size_t i=0; i<iter; ++i)
    {
        auto fw = model(input);
        Value<double> loss{0.0};
        for (size_t i{0}; i<fw.size(); ++i)
            loss = loss + pow(fw[i] - target[i], 2.0);
        loss.backward();

        if (i%(iter/100)==0)
        {
            for (const auto& v : model.get_parameters())
            {
                v->descend_grad(0.00001);
                v->zero_grad();
            }
        }

        if (i%(iter/5)==0)
            std::cout << "Output: " << model(input) << "\nLoss: " << loss << "\n";
    }
}

/*
void mnist_test()
{
    std::cout << "Loading MNIST data..." << std::endl;
    auto all_data = get_mnist_data<double>();
    auto& train_data = std::get<0>(all_data);
    auto& train_labels = std::get<1>(all_data);
    auto& test_data = std::get<2>(all_data);
    auto& test_labels = std::get<3>(all_data);
    std::cout << "Done!" << std::endl;

    std::cout << "Creating model..." << std::endl;
    MLP<double> model({784, 30, 10});
    std::cout << "Done!" << std::endl;

    std::cout << "Training model..." << std::endl;
    int batch_size = 50;
    int num_epochs = 3;
    double running_loss = 0.0;
    for (int epoch=0; epoch<num_epochs; ++epoch)
    {
        for (int i=0; i<train_data.size(); ++i)
        {
            auto loss = model.loss(train_data[i], train_labels[i]);
            loss.backward();
            running_loss += loss.get_data();
            if ((i+1) % batch_size == 0)
            {
                model.descend_grad(0.0001);
                model.zero_grad();
                std::cout << "Loss: " << running_loss / static_cast<double>(batch_size) << std::endl;
                running_loss = 0.0;
            }
        }
        std::cout << "Epoch " << epoch+1 << "/" << num_epochs << " complete." << std::endl;
    }
    std::cout << "Done!" << std::endl;

    std::cout << "Accuracy: " << evaluate_model(model, test_data, test_labels) << std::endl;
}
*/

int main()
{
    //set_seed();

    //sanity_check();

    MLP_test(100'000);
    //static_MLP_test(100'000);
    //mnist_test();

    //auto test = Neuron<double, 5>{};
    //std::cout << test({2.0, 3.0, -2.0, 3.0}) << std::endl;
    //std::cout << test.get_parameters() << std::endl;

    // auto l1 = StaticLayer<double, 1, 3>{};
    // std::cout << l1.get_parameters() << std::endl;

    // auto l2 = StaticLayer<double, 3, 1>{};
    // std::cout << l2.get_parameters() << std::endl;

    // auto m = StaticModule<decltype(l1), decltype(l2)>(std::move(l1), std::move(l2));

    // std::cout << m.get_parameters() << std::endl;

    // std::cout << m({3.0}) << std::endl;

    // for (int i=0; i<25; ++i)
    // {
    //     auto val = m({3.0})[0];
    //     val = val*val;
    //     val.backward();
    //     for (auto v : val.build_topo())
    //         v->descend_grad(0.01);
    //     val.zero_grad_all();
    //     std::cout << m({3.0}) << std::endl << std::endl;
    // }

    return 0;
}
