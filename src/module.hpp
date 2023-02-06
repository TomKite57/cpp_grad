
#pragma once

#include<memory>
#include<assert.h> 

#include "value.hpp"

// Interface for NN components
template <class T>
class Module
{
public:
    Module() = default;
    Module(Module&) = default;
    Module(Module&&) = default;
    virtual ~Module() = default;

    virtual std::vector<std::shared_ptr<Value<T>>> get_parameters() = 0;

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
    Value<T> _bias{0};

public:
    Neuron(size_t size): _size{size}
    {
        _weights = std::vector<Value<T>>(size, static_cast<T>(0));
        _bias = Value<T>(static_cast<T>(0));
    }
    Neuron(Neuron&) = default;
    Neuron(Neuron&&) = default;
    virtual ~Neuron() = default;

    virtual std::vector<std::shared_ptr<Value<T>>> get_parameters()
    {
        std::vector<std::shared_ptr<Value<T>>> rval = {std::make_shared<Value<T>>(_bias)};
        for (auto& w : _weights)
            rval.push_back(std::make_shared<Value<T>>(w));
        return rval;
    }

    virtual Value<T> operator()(std::vector<Value<T>>& input)
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

    virtual Value<T> operator()(std::vector<T>& input)
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
    Layer(size_t _size_in, size_t _size_out):
    _size_in{_size_in}, _size_out{_size_out}
    {
        for (size_t i=0; i<_size_out; ++i)
            _neurons.push_back(Neuron<T>(_size_in));
    }
    Layer(Layer&) = default;
    Layer(Layer&&) = default;
    virtual ~Layer() = default;

    virtual std::vector<std::shared_ptr<Value<T>>> get_parameters()
    {
        std::vector<std::shared_ptr<Value<T>>> rval;
        for (auto& n : _neurons)
        {
            auto temp = n.get_parameters();
            rval.insert(rval.end(), temp.begin(), temp.end());
        }
        return rval;
    }

    virtual std::vector<Value<T>> operator()(std::vector<Value<T>>& input)
    {
        std::vector<Value<T>> rval;
        for (auto& n : _neurons)
            rval.push_back(n(input));
        return rval;
    }

    virtual std::vector<Value<T>> operator()(std::vector<T>& input)
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
    MLP(std::vector<size_t> sizes)
    {
        for (size_t i=0; i<sizes.size()-1; ++i)
            _layers.push_back(Layer<T>(sizes[i], sizes[i+1]));
    };
    MLP(MLP&) = default;
    MLP(MLP&&) = default;
    virtual ~MLP() = default;

    virtual std::vector<std::shared_ptr<Value<T>>> get_parameters()
    {
        std::vector<std::shared_ptr<Value<T>>> rval;
        for (auto& l : _layers)
        {
            auto temp = l.get_parameters();
            rval.insert(rval.end(), temp.begin(), temp.end());
        }
        return rval;
    }

    virtual std::vector<Value<T>> operator()(std::vector<Value<T>>& input)
    {
        std::vector<Value<T>> rval = input;
        for (auto& l : _layers)
            rval = l(rval);
        return rval;
    }

    virtual std::vector<Value<T>> operator()(std::vector<T>& input)
    {
        std::vector<Value<T>> rval;
        
        for (auto& i : input)
            rval.push_back(Value<T>(i));

        for (auto& l : _layers)
            rval = l(rval);
        return rval;
    }

    virtual Value<T> loss(std::vector<T>& input, std::vector<T>& target)
    {
        auto output = operator()(input);
        Value<T> rval = Value<T>(0);
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
