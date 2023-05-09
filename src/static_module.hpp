#ifndef STATIC_MODULE_HPP
#define STATIC_MODULE_HPP

#include<iostream>
#include <tuple>
#include <utility>

#include "value.hpp"
#include "utils.hpp"

// Interface for NN components
template <class D, class T, size_t T_fan_in, size_t T_fan_out>
class StaticModuleBase
{
public:
    using _T = T;
    static constexpr size_t _T_fan_in   = T_fan_in;
    static constexpr size_t _T_fan_out  = T_fan_out;

public:
    StaticModuleBase() = default;
    StaticModuleBase(const StaticModuleBase&) = default;
    StaticModuleBase(StaticModuleBase&&) = default;
    StaticModuleBase& operator=(const StaticModuleBase&) = default;
    StaticModuleBase& operator=(StaticModuleBase&&) = default;
    ~StaticModuleBase() = default;

    //std::vector<std::shared_ptr<Value<T>>> get_parameters() const
    std::vector<const Value<T>*> get_parameters() const
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
class StaticNeuron: public StaticModuleBase<StaticNeuron<T, T_fan_in>, T, T_fan_in, 1>
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
    StaticNeuron(const bool& non_lin=true): _non_lin{non_lin}
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
    StaticNeuron(const StaticNeuron& other): _weights{other._weights}, _bias{other._bias} {}
    StaticNeuron(StaticNeuron&& other): _weights{std::move(other._weights)}, _bias{std::move(other._bias)} {}
    StaticNeuron& operator=(const StaticNeuron& other) { _weights = other._weights; _bias = other._bias; return *this; }
    StaticNeuron& operator=(StaticNeuron&& other) { _weights = std::move(other._weights); _bias = std::move(other._bias); return *this; }
    ~StaticNeuron() {}

    std::vector<const Value<T>*> get_parameters_impl() const
    {
        std::vector<const Value<T>*> rval;
        rval.push_back(&_bias);
        for (auto& w : _weights)
            rval.push_back(&w);
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
class StaticLayer: public StaticModuleBase<StaticLayer<T, T_fan_in, T_fan_out>, T, T_fan_in, T_fan_out>
{
public:
    using _T = T;
    static constexpr size_t _T_fan_in   = T_fan_in;
    static constexpr size_t _T_fan_out  = T_fan_out;

private:
    std::array<StaticNeuron<_T, _T_fan_in>, _T_fan_out> _neurons{};

public:
    StaticLayer()
    {
        // Might not be necessary, unless passing more params
        //for (size_t i=0; i<_T_fan_out; ++i)
        //    _neurons[i] = StaticNeuron<_T, _T_fan_in>{};
    }
    StaticLayer(const StaticLayer& other) { _neurons = other._neurons; };
    StaticLayer(StaticLayer&& other) { _neurons = std::move(other._neurons); };
    StaticLayer& operator=(const StaticLayer& other) { if (&other != this) _neurons = other._neurons; return *this; }
    StaticLayer& operator=(StaticLayer&& other) { _neurons = std::move(other._neurons); return *this; }
    ~StaticLayer() { };

    std::vector<const Value<T>*> get_parameters_impl() const
    {
        std::vector<const Value<T>*> rval;
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
class StaticModule: public StaticModuleBase<StaticModule<A, Rest...>, typename A::_T, A::_T_fan_in, StaticModule<Rest...>::_T_fan_out>
{
public:
    using _T = typename A::_T;
    static constexpr size_t _T_fan_in   = A::_T_fan_in;
    static constexpr size_t _T_fan_out  = StaticModule<Rest...>::_T_fan_out;

public:
    const A _a;
    const StaticModule<Rest...> _b;

    StaticModule(): _a{}, _b{} {};
    StaticModule(const A& a, const StaticModule<Rest...>& b): _a{a}, _b{b} {}
    StaticModule(const A&& a, const StaticModule<Rest...>&& b): _a{a}, _b{b} {}
    StaticModule(const StaticModule& other): _a{other._a}, _b{other._b} {}
    StaticModule(StaticModule&& other): _a{std::move(other._a)}, _b{std::move(other._b)} {}
    StaticModule& operator=(const StaticModule& other) { if (*this == other) return *this; _a = other._a; _b = other._b; return *this; }
    StaticModule& operator=(StaticModule&& other) { _a = std::move(other._a); _b = std::move(other._b); return *this; }
    ~StaticModule(){};

    std::vector<const Value<_T>*> get_parameters_impl() const
    {
        std::vector<const Value<_T>*> rval = _a.get_parameters();
        std::vector<const Value<_T>*> b_params = _b.get_parameters();
        rval.insert(rval.end(), b_params.begin(), b_params.end());
        return rval;
    }

    std::array<Value<_T>, _T_fan_out> operator_impl(const std::array<Value<_T>, _T_fan_in>& input) const
    {
        auto rval = _b(_a(input));
        return rval;
    }
};

template<class A>
class StaticModule<A>: public StaticModuleBase<StaticModule<A>, typename A::_T, A::_T_fan_in, A::_T_fan_out>
{
public:
    using _T = typename A::_T;
    static constexpr size_t _T_fan_in   = A::_T_fan_in;
    static constexpr size_t _T_fan_out  = A::_T_fan_out;

public:
    const A _a;

    StaticModule(): _a{} {};
    StaticModule(const A& a): _a{a} {};
    StaticModule(const A&& a): _a{a} {};
    StaticModule(const StaticModule& other): _a{other._a} {};
    StaticModule(StaticModule&& other): _a{std::move(other._a)} {};
    StaticModule& operator=(const StaticModule& other) { if (*this != other) _a = other._a; return *this; };
    StaticModule& operator=(StaticModule&& other) { _a = std::move(other._a); return *this; };
    ~StaticModule(){};

    std::vector<const Value<_T>*> get_parameters_impl() const
    {
        return _a.get_parameters();
    }

    std::array<Value<_T>, _T_fan_out> operator_impl(const std::array<Value<_T>, _T_fan_in>& input) const
    {
        auto rval = _a(input);
        return rval;
    }
};


#endif