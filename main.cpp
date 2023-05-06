
#include<iostream>
#include<cstdlib>
#include<tuple>

#include "src/value.hpp"
#include "src/new_module.hpp"
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

/*
void MLP_test()
{
    MLP<double> model({5, 5, 5, 5});

    std::vector<double> input = {4.7, 5.0, 5.2, 5.4, 5.6};
    std::vector<double> target = {1, 0, 0, 0, 0};

    for (size_t i=0; i<50; ++i)
    {
        auto loss = model.loss(input, target);
        loss.backward();
        model.descend_grad();
        model.zero_grad();
        std::cout << "Output: " << model(input) << "\nLoss: " << loss << "\n\n";
    }
}
*/

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
    set_seed();

    sanity_check();
    //MLP_test();
    //mnist_test();

    //auto test = Neuron<double, 5>{};
    //std::cout << test({2.0, 3.0, -2.0, 3.0}) << std::endl;
    //std::cout << test.get_parameters() << std::endl;

    auto l1 = Layer<double, 1, 3>{};
    std::cout << l1.get_parameters() << std::endl;

    auto l2 = Layer<double, 3, 1>{};
    std::cout << l2.get_parameters() << std::endl;

    //auto m = Module<decltype(l1), decltype(l2)>(l1, l2);
//
    //std::cout << m.get_parameters() << std::endl;
//
    std::cout << l2(l1({3.0})) << std::endl;
    //std::cout << m({3.0}) << std::endl;

    return 0;
}
