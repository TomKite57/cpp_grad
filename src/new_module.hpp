#ifndef NEW_MODULE_HPP
#define NEW_MODULE_HPP

#include<iostream>
#include <tuple>
#include <utility>

#include "value.hpp"

// template<typename T>
// concept is_module = requires(T)
// {
//     T::type;
//     T::fan_in;
//     T::fan_out;
// };


template <class T>
T get_random_number(const T& min, const T& max)
{
    return static_cast<T>(rand()) / static_cast<T>(RAND_MAX) * (max - min) + min;
}

// Interface for NN components
template <class D, class T, size_t T_fan_in, size_t T_fan_out>
class ModuleBase
{
public:
    using _T = T;
    static constexpr size_t _T_fan_in   = T_fan_in;
    static constexpr size_t _T_fan_out  = T_fan_out;

public:
    ModuleBase() = default;
    ModuleBase(const ModuleBase&) = default;
    ModuleBase(ModuleBase&&) = default;
    ModuleBase& operator=(const ModuleBase&) = default;
    ModuleBase& operator=(ModuleBase&&) = default;
    ~ModuleBase() = default;

    std::vector<std::shared_ptr<Value<T>>> get_parameters() const
    {
        return static_cast<const D*>(this)->get_parameters_impl();
    };

    void zero_grad()
    {
        for (auto& node : get_parameters())
            node->zero_grad();
    }

    std::array<Value<_T>, _T_fan_out> operator()(const std::array<Value<_T>, _T_fan_in>& input) const
    {
        return static_cast<const D*>(this)->operator_impl(input);
    };
};


template <class T, size_t T_fan_in>
class Neuron: public ModuleBase<Neuron<T, T_fan_in>, T, T_fan_in, 1>
{
public:
    using _T = T;
    static constexpr size_t _T_fan_in   = T_fan_in;
    static constexpr size_t _T_fan_out  = 1;

private:
    bool _non_lin;
    std::array<Value<T>, T_fan_in> _weights{};
    Value<T> _bias{static_cast<T>(0)};

public:
    Neuron(const bool& non_lin=true): _non_lin{non_lin}
    {
        constexpr T max = static_cast<T>(1);
        constexpr T min = static_cast<T>(-1);
        for (size_t i=0; i<T_fan_in; ++i)
        {
            // TODO Check Andrej's video for correct coefficient
            _weights[i] = Value<T>(get_random_number(min, max)/static_cast<T>(T_fan_in));
        }
        _bias = Value<T>(get_random_number(static_cast<T>(-1), static_cast<T>(1)));
    }
    Neuron(const Neuron& other): _weights{other._weights}, _bias{other._bias} {}
    Neuron(Neuron&& other): _weights{std::move(other._weights)}, _bias{std::move(other._bias)} {}
    Neuron& operator=(const Neuron& other) { _weights = other._weights; _bias = other._bias; return *this; }
    Neuron& operator=(Neuron&& other) { _weights = std::move(other._weights); _bias = std::move(other._bias); return *this; }
    ~Neuron() {}

    std::vector<std::shared_ptr<Value<T>>> get_parameters_impl() const
    {
        std::vector<std::shared_ptr<Value<T>>> rval = {std::make_shared<Value<T>>(_bias)};
        for (auto& w : _weights)
            rval.push_back(std::make_shared<Value<T>>(w));
        return rval;
    }

    std::array<Value<T>, 1> operator_impl(const std::array<Value<T>, T_fan_in>& input) const
    {
        std::array<Value<T>, 1> rval{_bias};

        for (size_t i=0; i<T_fan_in; ++i)
        {
            auto new_temp = input[i] * _weights[i];
            rval[0] = rval[0] + new_temp;
        }
        return rval;
    }
};

template <class T, size_t T_fan_in, size_t T_fan_out>
class Layer: public ModuleBase<Layer<T, T_fan_in, T_fan_out>, T, T_fan_in, T_fan_out>
{
public:
    using _T = T;
    static constexpr size_t _T_fan_in   = T_fan_in;
    static constexpr size_t _T_fan_out  = T_fan_out;

private:
    std::array<Neuron<_T, _T_fan_in>, _T_fan_out> _neurons{};

public:
    Layer()
    {
        // Might not be necessary, unless passing more params
        //for (size_t i=0; i<_T_fan_out; ++i)
        //    _neurons[i] = Neuron<_T, _T_fan_in>{};
    }
    Layer(const Layer& other) { _neurons = other._neurons; };
    Layer(Layer&& other) { _neurons = std::move(other._neurons); };
    Layer& operator=(const Layer& other) { if (&other != this) _neurons = other._neurons; return *this; }
    Layer& operator=(Layer&& other) { _neurons = std::move(other._neurons); return *this; }
    ~Layer() { };

    std::vector<std::shared_ptr<Value<_T>>> get_parameters_impl() const
    {
        std::vector<std::shared_ptr<Value<_T>>> rval;
        for (auto& n : _neurons)
        {
            auto temp = n.get_parameters();
            rval.insert(rval.end(), temp.begin(), temp.end());
        }
        return rval;
    }

    std::array<Value<_T>, _T_fan_out> operator_impl(const std::array<Value<_T>, _T_fan_in>& input) const
    {
        std::array<Value<_T>, _T_fan_out> rval;
        for (size_t i{0}; i<_T_fan_out; ++i)
        {
            rval[i] = _neurons[i](input)[0];
        }
        return rval;
    }
};


template<class A, class... Rest>
class Module: public ModuleBase<Module<A, Rest...>, typename A::_T, A::_T_fan_in, Module<Rest...>::_T_fan_out>
{
public:
    using _T = typename A::_T;
    static constexpr size_t _T_fan_in   = A::_T_fan_in;
    static constexpr size_t _T_fan_out  = Module<Rest...>::_T_fan_out;

public:
    const A _a;
    const Module<Rest...> _b;

    Module() = delete;
    Module(const A& a, const Module<Rest...>& b): _a{a}, _b{b} {}
    Module(const A&& a, const Module<Rest...>&& b): _a{a}, _b{b} {}
    Module(const Module& other): _a{other._a}, _b{other._b} {}
    Module(Module&& other): _a{std::move(other._a)}, _b{std::move(other._b)} {}
    Module& operator=(const Module& other) { if (*this == other) return *this; _a = other._a; _b = other._b; return *this; }
    Module& operator=(Module&& other) { _a = std::move(other._a); _b = std::move(other._b); return *this; }
    ~Module(){};

    std::vector<std::shared_ptr<Value<_T>>> get_parameters_impl() const
    {
        std::vector<std::shared_ptr<Value<_T>>> rval = _a.get_parameters();
        std::vector<std::shared_ptr<Value<_T>>> b_params = _b.get_parameters();
        rval.insert(rval.end(), b_params.begin(), b_params.end());
        return rval;
    }

    std::array<Value<_T>, _T_fan_out> operator_impl(const std::array<Value<_T>, _T_fan_in>& input) const
    {
        auto rval = _b(_a(input));
        return rval;
    }

    std::array<Value<_T>, _T_fan_out> operator_impl(const std::array<Value<_T>, _T_fan_in>&& input) const
    {
        auto rval = _b(_a(input));
        return rval;
    }
};

template<class A>
class Module<A>: public ModuleBase<Module<A>, typename A::_T, A::_T_fan_in, A::_T_fan_out>
{
public:
    using _T = typename A::_T;
    static constexpr size_t _T_fan_in   = A::_T_fan_in;
    static constexpr size_t _T_fan_out  = A::_T_fan_out;

public:
    const A _a;

    Module() = delete;
    Module(const A& a): _a{a} {};
    Module(const A&& a): _a{a} {};
    Module(const Module& other): _a{other._a} {};
    Module(Module&& other): _a{std::move(other._a)} {};
    Module& operator=(const Module& other) { if (*this != other) _a = other._a; return *this; };
    Module& operator=(Module&& other) { _a = std::move(other._a); return *this; };
    ~Module(){};

    std::vector<std::shared_ptr<Value<_T>>> get_parameters_impl() const
    {
        return _a.get_parameters();
    }

    std::array<Value<_T>, _T_fan_out> operator_impl(const std::array<Value<_T>, _T_fan_in>& input) const
    {
        auto rval = _a(input);
        return rval;
    }

    std::array<Value<_T>, _T_fan_out> operator_impl(const std::array<Value<_T>, _T_fan_in>&& input) const
    {
        auto rval = _a(input);
        return rval;
    }
};


#endif