
#pragma once

#include<iostream>
#include<memory>
#include<vector>
#include<assert.h> 
#include<cstdlib>

#include "value.hpp"

template <class T>
T get_random_number(const T& min, const T& max)
{
    return static_cast<T>(rand()) / static_cast<T>(RAND_MAX) * (max - min) + min;
}

// Interface for NN components
template <class T>
class Module
{
public:
    Module() = default;
    Module(const Module&) = default;
    Module(Module&&) = default;
    virtual ~Module() = default;

    virtual std::vector<std::shared_ptr<Value<T>>> get_parameters() const = 0;

    void zero_grad()
    {
        for (auto& node : get_parameters())
            node->zero_grad();
    }
};

template <class T>
class Neuron: public Module<T>
{
private:
    size_t _size;
    std::vector<Value<T>> _weights;
    Value<T> _bias{static_cast<T>(0)};

public:
    Neuron(const size_t& size): _size{size}
    {
        for (size_t i=0; i<size; ++i)
            _weights.push_back(Value<T>(get_random_number(static_cast<T>(-1.0), static_cast<T>(1.0))));
        _bias = Value<T>(get_random_number(static_cast<T>(-1.0), static_cast<T>(1.0)));
    }
    Neuron(const Neuron& other) { _size = other._size; _weights = other._weights; _bias = other._bias; }
    Neuron(Neuron&& other) { _size = other._size; _weights = std::move(other._weights); _bias = std::move(other._bias); }
    Neuron& operator=(const Neuron& other) { _size = other._size; _weights = other._weights; _bias = other._bias; return *this; }
    Neuron& operator=(Neuron&& other) { _size = other._size; _weights = std::move(other._weights); _bias = std::move(other._bias); return *this; }
    ~Neuron() { _weights.clear(); }

    std::vector<std::shared_ptr<Value<T>>> get_parameters() const
    {
        std::vector<std::shared_ptr<Value<T>>> rval = {std::make_shared<Value<T>>(_bias)};
        for (auto& w : _weights)
            rval.push_back(std::make_shared<Value<T>>(w));
        return rval;
    }

    Value<T> operator()(const std::vector<Value<T>>& input) const
    {
        assert(input.size() == _size);

        Value<T> rval = _bias;
        for (size_t i=0; i<_size; ++i)
        {
            auto new_temp = input[i] * _weights[i];
            rval = rval + new_temp;
        }
        return rval;
    }

    Value<T> operator()(const std::vector<T>& input) const
    {
        assert(input.size() == _size);

        Value<T> rval = _bias;
        for (size_t i=0; i<_size; ++i)
        {
            Value<T> temp(input[i]);
            auto new_temp = temp * _weights[i];
            rval = rval + new_temp;
        }
        return rval;
    }
};


template <class T>
class Layer: public Module<T>
{
private:
    size_t _size_in, _size_out;
    std::vector<Neuron<T>> _neurons;

public:
    Layer(const size_t& _size_in, const size_t& _size_out):
    _size_in{_size_in}, _size_out{_size_out}
    {
        for (size_t i=0; i<_size_out; ++i)
            _neurons.push_back(Neuron<T>(_size_in));
    }
    Layer(const Layer& other)
    {
        _size_in = other._size_in;
        _size_out = other._size_out;
        _neurons = other._neurons;
    }
    Layer(Layer&& other)
    {
        _size_in = other._size_in;
        _size_out = other._size_out;
        _neurons = std::move(other._neurons);
    }
    ~Layer() { _neurons.clear(); };

    std::vector<std::shared_ptr<Value<T>>> get_parameters() const
    {
        std::vector<std::shared_ptr<Value<T>>> rval;
        for (auto& n : _neurons)
        {
            auto temp = n.get_parameters();
            rval.insert(rval.end(), temp.begin(), temp.end());
        }
        return rval;
    }

    std::vector<Value<T>> operator()(const std::vector<Value<T>>& input) const
    {
        std::vector<Value<T>> rval;
        for (auto& n : _neurons)
        {
            rval.push_back(n(input));
        }
        return rval;
    }

    std::vector<Value<T>> operator()(const std::vector<T>& input) const
    {
        std::vector<Value<T>> rval;
        for (auto& n : _neurons)
            rval.push_back(n(input));
        return rval;
    }
};

template <class T>
class MLP: public Module<T>
{
private:
    std::vector<Layer<T>> _layers;

public:
    MLP(const std::vector<size_t>& sizes)
    {
        for (size_t i=0; i<sizes.size()-1; ++i)
            _layers.push_back(Layer<T>(sizes[i], sizes[i+1]));
    }
    MLP(const MLP&) = delete;
    MLP(MLP&&) = delete;
    ~MLP() { _layers.clear(); }

    std::vector<std::shared_ptr<Value<T>>> get_parameters() const
    {
        std::vector<std::shared_ptr<Value<T>>> rval;
        for (auto& l : _layers)
        {
            auto temp = l.get_parameters();
            rval.insert(rval.end(), temp.begin(), temp.end());
        }
        return rval;
    }

    std::vector<Value<T>> operator()(const std::vector<Value<T>>& input) const
    {
        std::vector<Value<T>> rval = input;
        for (auto& l : _layers)
            rval = l(rval);
        return rval;
    }

    std::vector<Value<T>> operator()(const std::vector<T>& input) const
    {
        std::vector<Value<T>> rval;
        
        for (auto& i : input)
            rval.push_back(Value<T>(i));

        for (auto& l : _layers)
            rval = l(rval);
        return rval;
    }

    Value<T> loss(const std::vector<T>& input, const std::vector<T>& target) const
    {
        auto output = operator()(input);

        Value<T> rval = Value<T>(static_cast<T>(0));
        for (size_t i=0; i<output.size(); ++i)
        {
            auto target_value = Value<T>(target[i]);
            auto diff = output[i] - target_value;
            auto temp_loss = pow(diff, static_cast<T>(2));
            rval = rval + temp_loss;
        }
        return rval;
    }

    Value<T> loss(const std::vector<Value<T>>& input, const std::vector<T>& target) const
    {
        std::vector<Value<T>> output = operator()(input);

        Value<T> rval(static_cast<T>(0));
        for (size_t i=0; i<output.size(); ++i)
        {
            auto target_value = Value<T>(target[i]);
            auto diff = output[i] - target_value;
            auto temp_loss = pow(diff, static_cast<T>(2));
            rval = rval + temp_loss;
        }
        return rval;
    }
};
