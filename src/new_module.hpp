#ifndef NEW_MODULE_HPP
#define NEW_MODULE_HPP

#include<iostream>

#include "value.hpp"

template <class T>
T get_random_number(const T& min, const T& max)
{
    return static_cast<T>(rand()) / static_cast<T>(RAND_MAX) * (max - min) + min;
}

// Interface for NN components
template <class T, size_t fan_in, size_t fan_out>
class Module
{
public:
    // fan variables for use in class
    static constexpr size_t _fan_in = fan_in;
    static constexpr size_t _fan_out = fan_out;

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

    virtual std::array<Value<T>, fan_out> operator()(const std::array<Value<T>, fan_in>&) const = 0;
};

template <class T, size_t fan_in>
class Neuron: public Module<T, fan_in, 1>
{
private:
    bool _non_lin;
    std::array<Value<T>, fan_in> _weights{};
    Value<T> _bias{static_cast<T>(0)};

public:
    Neuron(const bool& non_lin=true): _non_lin{non_lin}
    {
        constexpr T max = static_cast<T>(1);
        constexpr T min = static_cast<T>(-1);
        for (size_t i=0; i<fan_in; ++i)
        {
            // TODO Check Andrej's video for correct coefficient
            _weights[i] = Value<T>(get_random_number(min, max)/static_cast<T>(fan_in));
        }
        _bias = Value<T>(get_random_number(static_cast<T>(-1), static_cast<T>(1)));
    }
    Neuron(const Neuron& other) { _weights = other._weights; _bias = other._bias; }
    Neuron(Neuron&& other) { _weights = std::move(other._weights); _bias = std::move(other._bias); }
    Neuron& operator=(const Neuron& other) { _weights = other._weights; _bias = other._bias; return *this; }
    Neuron& operator=(Neuron&& other) { _weights = std::move(other._weights); _bias = std::move(other._bias); return *this; }
    ~Neuron() {}

    std::vector<std::shared_ptr<Value<T>>> get_parameters() const
    {
        std::vector<std::shared_ptr<Value<T>>> rval = {std::make_shared<Value<T>>(_bias)};
        for (auto& w : _weights)
            rval.push_back(std::make_shared<Value<T>>(w));
        return rval;
    }

    std::array<Value<T>, 1> operator()(const std::array<Value<T>, fan_in>& input) const
    {
        std::array<Value<T>, 1> rval{_bias};

        for (size_t i=0; i<fan_in; ++i)
        {
            auto new_temp = input[i] * _weights[i];
            rval[0] = rval[0] + new_temp;
        }
        return rval;
    }
};


template <class T, size_t fan_in, size_t fan_out>
class Layer: public Module<T, fan_in, fan_out>
{
private:
    std::array<Neuron<T, fan_in>, fan_out> _neurons{};

public:
    Layer()
    {
        // Might not be necessary, unless passing more params
        for (size_t i=0; i<fan_out; ++i)
            _neurons[i] = Neuron<T, fan_in>{};
    }
    Layer(const Layer& other)
    {
        _neurons = other._neurons;
    }
    Layer(Layer&& other)
    {
        _neurons = std::move(other._neurons);
    }
    ~Layer() { };

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

    std::array<Value<T>, fan_out> operator()(const std::array<Value<T>, fan_in>& input) const
    {
        std::array<Value<T>, fan_out> rval;
        for (size_t i{0}; i<fan_out; ++i)
            rval[i] = _neurons[i](input)[0];
        return rval;
    }
};

template <class T, size_t fan_in_1, size_t fan_out_1, size_t fan_in_2, size_t fan_out_2>
constexpr bool are_compatible_layers(const Layer<T, fan_in_1, fan_out_1>& l1, const Layer<T, fan_in_2, fan_out_2>& l2) {
    return fan_out_1 == fan_in_2;
}

template <class... Args>
class Pipeline;

template <class T, size_t fan_in, size_t fan_out, class... Args>
class Pipeline<Layer<T, fan_in, fan_out>, Args...> {
public:
    static constexpr size_t input_size = fan_in;
    static constexpr size_t output_size = Layer<T, fan_in, fan_out>::fan_out;

    static_assert(are_compatible_layers(std::declval<Layer<T, fan_in, fan_out>>(), std::declval<typename Pipeline<Args...>::head>()), "Layers are not compatible");

    using head = Layer<T, fan_in, fan_out>;
    using tail = Pipeline<Args...>;
};

template <>
class Pipeline<> {
public:
    static constexpr size_t input_size = 0;
    static constexpr size_t output_size = 0;

    using head = void;
    using tail = void;
};

template <class... Args>
class Pipeline {
public:
    Pipeline(Args... args)
        : layers(args...) {};

    // Other functions and member variables...

private:
    std::tuple<Args...> layers;
};

/*
template <class T, size_t... sizes>
class my_class: public Module<T, std::get<0>(std::make_tuple(sizes...)), std::get<sizeof...(sizes)-1>(std::make_tuple(sizes...))>
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

    void descend_grad(const T& learning_rate=static_cast<T>(0.01))
    {
        for (const auto& p : get_parameters())
            p->descend_grad(learning_rate);
    }

    void zero_grad()
    {
        for (const auto& p : get_parameters())
            p->zero_grad();
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
*/


#endif