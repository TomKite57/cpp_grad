
#pragma once

#include<iostream>
#include<cmath>
#include<utility>
#include<functional>
#include<set>
#include<memory>

const std::function<void()> do_nothing = [](){return;};

// Forward declarations
template<class T> class Value;
template<class T> std::vector<Value<T>*> topological_sort(Value<T>&);

// Central Value class
template <class T>
class Value
{
    template <class C>
    friend std::ostream& operator<<(std::ostream& os, const Value<C>& val)
    {
        os << "Value(" << val.get_data() << ", " << val.get_grad() << ")";
        return os;
    }

    template <class C>
    friend Value<C> pow(Value<C>& val, C exp)
    {
        auto out = Value(std::pow(val._data, exp), {std::make_shared<Value<C>>(val),});

        auto _back = [&]()
        {
            val._grad += (exp * std::pow(val._data, exp- static_cast<T>(1))) * out._grad;
        };
        out._backward = _back;

        return out;
    }

    template <class C>
    friend Value<C> operator+(C num, Value<C>& val) {return val + num;}

    template <class C>
    friend Value<C> operator-(C num, Value<C>& val) {return val - num;}

    template <class C>
    friend Value<C> operator*(C num, Value<C>& val) {return val * num;}

    template <class C>
    friend Value<C> operator/(C num, Value<C>& val) {return val / num;}

private:
    T _data{0};
    T _grad{0};
    std::vector<std::shared_ptr<Value<T>>> _parents;
    std::function<void()> _backward = do_nothing;


    Value(T data, std::vector<std::shared_ptr<Value<T>>> parents):
    _data{data}, _parents{parents}
    {}

public:
    Value(T data): _data{data} {}
    Value(Value&) = default;
    Value(Value&&) = default;
    ~Value() = default;

    Value<T>& operator=(Value<T>& other) = default;
    Value<T>& operator=(Value<T>&& other) = default;

    T get_data() const { return _data; }
    T get_grad() const { return _grad; }
    std::vector<std::shared_ptr<Value<T>>> get_parent_ptrs() { return _parents; }

    void zero_grad(){ _grad = static_cast<T>(0); }

    void backward()
    {
        auto order = build_topo(this);

        // Set dx/dx=1
        _grad = static_cast<T>(1);
        for (auto n=order.rbegin(); n!=order.rend(); ++n)
        {
            (*n)->_backward();
        }
    }

    Value<T> operator+(Value<T>& other)
    {
        auto out = Value<T>(
            _data + other._data,
            {std::make_shared<Value<T>>(*this), std::make_shared<Value<T>>(other)}
        );

        auto _back = [&]()
        {
            _grad += out._grad;
            other._grad += out._grad;
        };
        out._backward = _back;

        return out;
    }

    Value<T> operator+(T other)
    {
        auto temp = Value<T>(other);
        return operator+(temp);
    }

    Value<T> operator-(Value<T>& other)
    {
        auto out = Value<T>(
            _data - other._data,
            {std::make_shared<Value<T>>(*this), std::make_shared<Value<T>>(other)}
        );

        auto _back = [&]()
        {
            _grad += out._grad;
            other._grad += out._grad;
        };
        out._backward = _back;

        return out;
    }

    Value<T> operator-(T other)
    {
        auto temp = Value<T>(other);
        return operator-(temp);
    }

    Value<T> operator*(Value<T>& other)
    {
        auto out = Value<T>(
            _data * other._data,
            {std::make_shared<Value<T>>(*this), std::make_shared<Value<T>>(other)}
        );

        auto _back = [&]()
        {
            _grad += other._data * out._grad;
            other._grad += _data * out._grad;
        };
        out._backward = _back;

        return out;
    }

    Value<T> operator*(T other)
    {
        auto temp = Value<T>(other);
        return operator*(temp);
    }

    Value<T> operator/(Value<T>& other)
    {
        auto temp = pow(other, static_cast<T>(-1));
        return operator*(temp);
    }

    Value<T> operator/(T other)
    {
        auto temp = Value<T>(other);
        return operator/(temp);
    }

    Value<T> operator-()
    {
        return operator*(static_cast<T>(-1));
    }
};

template<class T>
std::vector<Value<T>*> build_topo(Value<T>* root)
{
    std::vector<Value<T>*> rval;
    std::set<Value<T>*> visited;
    _build_topo(root, visited, rval);
    return rval;
}

// [TODO] Make class for encapsulation
template<class T>
void _build_topo(Value<T>* node, std::set<Value<T>*>& visited, std::vector<Value<T>*>& order)
{
    if (visited.find(node) != visited.end())
        return;

    visited.insert(node);
    for (auto& par_ptr : node->get_parent_ptrs())
        _build_topo(par_ptr.get(), visited, order);
    order.push_back(node);
}
