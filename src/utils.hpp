
#ifndef UTILS_HPP
#define UTILS_HPP

#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<cstdlib>
#include<tuple>

// Vector printout
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "Vector(";
    for (auto ptr=v.begin(); ptr!=v.end(); ++ptr)
    {
        os << *ptr;
        if (ptr != v.end()-1)
            os << ", ";
    }
    os << ")";
    return os;
}

void set_seed()
{
    srand((unsigned) time(NULL));
}

template <class T>
std::vector<std::vector<T>> read_mnist(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cout << "Error opening file: " << filename << "." << std::endl;
        throw;
    }

    std::vector<std::vector<T>> data;
    for (std::string str; std::getline(file, str);)
    {
        std::vector<T> row;
        std::stringstream ss(str);
        for (T i; ss >> i;)
            row.push_back(i);

        data.push_back(row);
        row.clear();
    }

    return data;
}

template <class T>
std::vector<T> make_label_vector(T label, size_t size)
{
    std::vector<T> vec(size, static_cast<T>(0));
    vec[static_cast<size_t>(label)] = static_cast<T>(1);
    return vec;
}

template <class T>
std::vector<T> make_label_vector(std::vector<T> label, size_t size)
{
    assert(label.size() == 1);

    std::vector<T> vec(size, static_cast<T>(0));
    vec[static_cast<size_t>(label[0])] = static_cast<T>(1);
    return vec;
}


template <class T>
std::tuple<
    std::vector<std::vector<T>>,
    std::vector<std::vector<T>>,
    std::vector<std::vector<T>>,
    std::vector<std::vector<T>>
> get_mnist_data()
{
    std::vector<std::vector<T>> train_images = read_mnist<T>("data/mnist/mnist_train_images.txt");
    std::vector<std::vector<T>> train_labels = read_mnist<T>("data/mnist/mnist_train_labels.txt");
    std::vector<std::vector<T>> test_images = read_mnist<T>("data/mnist/mnist_test_images.txt");
    std::vector<std::vector<T>> test_labels = read_mnist<T>("data/mnist/mnist_test_labels.txt");

    for (auto& row : train_images)
        for (auto& val : row)
            val = val / static_cast<T>(255) - 0.5;

    for (auto& row : test_images)
        for (auto& val : row)
            val = val / static_cast<T>(255) - 0.5;
    
    for (auto& row : train_labels)
        row = make_label_vector(row[0], 10);
    
    for (auto& row : test_labels)
        row = make_label_vector(row[0], 10);

    return std::make_tuple(train_images, train_labels, test_images, test_labels);
}

template <class T>
T evaluate_model(const MLP<T>& model, const std::vector<std::vector<T>>& test_data, const std::vector<std::vector<T>>& test_labels)
{
    assert(test_data.size() == test_labels.size());

    T correct = 0;
    for (size_t i=0; i<test_data.size(); ++i)
    {
        auto prediction = model(test_data[i]);
        auto max = std::max_element(prediction.begin(), prediction.end());
        auto max_index = std::distance(prediction.begin(), max);
        if (test_labels[i][max_index] == static_cast<T>(1))
            ++correct;
    }
    return correct / static_cast<T>(test_data.size());
}


#endif