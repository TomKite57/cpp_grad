
#pragma once

#include<iostream>
#include<cmath>
#include<utility>
#include<functional>
#include<set>
#include<memory>

std::function<void()> do_nothing = [](){return;};

// Forward declarations
template<class T> class _Value;
template<class T> class Value;
template<class T> std::vector<_Value<T>*> build_topo(_Value<T>*);
template<class T> void _build_topo(_Value<T>*, std::set<_Value<T>*>&, std::vector<_Value<T>*>&);

// A "Hidden" value class which can only be heap allocated. Will be accessed through the proxy class

template <class T>
class _Value
{
    template <class C> friend class Value;

    template <class C>
    friend std::ostream& operator<<(std::ostream& os, _Value<C>& val)
    {
        os << "Value(" << val.get_data() << ", " << val.get_grad() << ")";
        return os;
    }

private:
    T _data{0};
    T _grad{0};
    std::vector<std::shared_ptr<_Value<T>>> _parents;
    std::function<void()> _backward = do_nothing;

public:
    _Value(T data, std::vector<std::shared_ptr<_Value<T>>> parents):
    _data{data}, _parents{parents}
    {}

    // Constructor and destructor
    _Value(T data): _data{data} {}
    ~_Value() = default;

    // Copy and move constructors
    _Value(_Value&) = delete;
    _Value(_Value&&) = delete;

    // Copy and move assignment operators
    _Value<T>& operator=(_Value<T>& other) = delete;
    _Value<T>& operator=(_Value<T>&& other) = delete;

    // Getters (Note reference return type however)
    T& get_data() { return _data; }
    T& get_grad() { return _grad; }
    std::vector<std::shared_ptr<_Value<T>>>& get_parent_ptrs() { return _parents; }

    // Setters
    void zero_grad(){ _grad = static_cast<T>(0); }
    void set_backward(std::function<void()> func){ _backward = func; }

    // Topological sort
    std::vector<_Value<T>*> build_topo()
    {
        std::vector<_Value<T>*> rval;
        std::set<_Value<T>*> visited;

        // Build hidden function in this scope
        std::function<void(_Value<T>*, std::set<_Value<T>*>&, std::vector<_Value<T>*>&)> _build_topo;
        _build_topo = [&_build_topo](_Value<T>* node, std::set<_Value<T>*>& visited, std::vector<_Value<T>*>& order)
        {
            if (visited.find(node) != visited.end())
                return;

            visited.insert(node);
            for (auto& par_ptr : node->get_parent_ptrs())
                _build_topo(par_ptr.get(), visited, order);
            order.push_back(node);
        };

        _build_topo(this, visited, rval);
        return rval;
    }

    // Backpropagation
    void backward()
    {
        auto order = build_topo();

        // Set dx/dx=1
        _grad = static_cast<T>(1);
        for (auto n=order.rbegin(); n!=order.rend(); ++n)
            (*n)->_backward();
    }
};

// Central Value class
template <class T>
class Value
{
    template <class C>
    friend std::ostream& operator<<(std::ostream& os, Value<C>& val)
    {
        os << "Value(" << val.get_data() << ", " << val.get_grad() << ")";
        return os;
    }

    template <class C>
    friend Value<C> pow(Value<C>& val, C exp)
    {
        auto out = Value(std::pow(val.get_data(), exp), {std::make_shared<Value<C>>(val),});

        std::shared_ptr<_Value<T>>& val_ptr = val._ptr;
        std::shared_ptr<_Value<T>>& out_ptr = out._ptr;

        auto _back = [=]()
        {
            val.get_grad() += (exp * std::pow(val_ptr->get_data(), exp- static_cast<T>(1))) * out_ptr->get_grad();
        };
        out.set_backward(_back);

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
    std::shared_ptr<_Value<T>> _ptr = nullptr;

    Value(T data, std::vector<std::shared_ptr<_Value<T>>> parents) { _ptr = std::make_shared<_Value<T>>(data, parents); }

public:
    // Constructors and destructors
    Value(T data) { _ptr = std::make_shared<_Value<T>>(data); }
    ~Value() { _ptr = nullptr; };

    // Copy and move constructors
    Value(Value& other) { _ptr = other._ptr; }
    Value(Value&& other) { _ptr = other._ptr; other._ptr = nullptr; }

    // Copy and move assignment operators
    Value<T>& operator=(Value<T>& other) { if (&other!=this) _ptr = other._ptr; return *this; }
    Value<T>& operator=(Value<T>&& other) { _ptr = other._ptr; other._ptr = nullptr; return *this; }

    // Transparency to the _Value class
    T& get_data() { return _ptr->get_data(); }
    T& get_grad() { return _ptr->get_grad(); }
    std::vector<std::shared_ptr<_Value<T>>>& get_parent_ptrs() { return _ptr->get_parent_ptrs(); }
    void zero_grad(){ _ptr->zero_grad(); }
    void set_backward(std::function<void()> func){ _ptr->set_backward(func); }
    void backward(){ _ptr->backward(); }
    std::vector<_Value<T>*> build_topo(){ return _ptr->build_topo(); }

    // [TODO] DEBUG ACCESSOR ONLY
    std::shared_ptr<_Value<T>> get_ptr() { return _ptr; }

    // Arithmetic operators
    Value<T> operator+(Value<T>& other)
    {
        Value<T> out(
            get_data() + other.get_data(),
            {get_ptr(), other.get_ptr()}
        );

        std::shared_ptr<_Value<T>>& this_ptr = _ptr;
        std::shared_ptr<_Value<T>>& other_ptr = other._ptr;
        std::shared_ptr<_Value<T>>& out_ptr = out._ptr;

        auto _back = [=]()
        {
            this_ptr->_grad += out_ptr->_grad;
            other_ptr->_grad += out_ptr->_grad;
        };
        out.set_backward(_back);

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
            get_data() - other.get_data(),
            {get_ptr(), other.get_ptr()}
        );

        std::shared_ptr<_Value<T>>& this_ptr = _ptr;
        std::shared_ptr<_Value<T>>& other_ptr = other._ptr;
        std::shared_ptr<_Value<T>>& out_ptr = out._ptr;

        auto _back = [=]()
        {
            this_ptr->get_grad() += out_ptr->get_grad();
            other_ptr->get_grad() += out_ptr->get_grad();
        };
        out.set_backward(_back);

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
            get_data() * other.get_data(),
            {get_ptr(), other.get_ptr()}
        );

        std::shared_ptr<_Value<T>>& this_ptr = _ptr;
        std::shared_ptr<_Value<T>>& other_ptr = other._ptr;
        std::shared_ptr<_Value<T>>& out_ptr = out._ptr;

        auto _back = [&]()
        {
            this_ptr->get_grad() += other_ptr->get_data() * out_ptr->get_grad();
            other_ptr->get_grad() += this_ptr->get_data() * out_ptr->get_grad();
        };
        out.set_backward(_back);

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
