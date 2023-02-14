#ifndef VALUE_HPP
#define VALUE_HPP

#include<iostream>
#include<cmath>
#include<vector>
#include<utility>
#include<functional>
#include<set>
#include<memory>

const std::function<void()> do_nothing = [](){return;};

// Forward declarations
template<class T> class _Value;
template<class T> class Value;

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
    T _data{static_cast<T>(0)};
    T _grad{static_cast<T>(0)};
    std::vector<std::shared_ptr<_Value<T>>> _parents;
    std::function<void()> _backward = do_nothing;

public:
    _Value(const T& data, const std::vector<std::shared_ptr<_Value<T>>>& parents):
    _data{data}, _parents{parents}
    {}

    // Constructor and destructor
    _Value(const T& data): _data{data} {}
    ~_Value() = default;

    // Copy and move constructors
    _Value(const _Value&) = delete;
    _Value(_Value&&) = delete;

    // Copy and move assignment operators
    _Value<T>& operator=(const _Value<T>& other) = delete;
    _Value<T>& operator=(_Value<T>&& other) = delete;

    // Getters (Note reference return type however)
    const T& get_data() const { return _data; }
    const T& get_grad() const { return _grad; }
    T& get_data() { return _data; }
    T& get_grad() { return _grad; }
    const std::vector<std::shared_ptr<_Value<T>>>& get_parent_ptrs() const { return _parents; }

    // Setters
    void zero_grad() { _grad = static_cast<T>(0); }
    void zero_grad_all()
    {
        auto order = build_topo();
        for (auto n=order.rbegin(); n!=order.rend(); ++n)
            (*n)->zero_grad();
    }
    void set_backward(const std::function<void()>& func) { _backward = func; }

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

    void descend_grad(const T& learning_rate)
    {
        _data -= learning_rate * _grad;
    }
};

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
    friend Value<C> pow(const Value<C>& val,  const C& exp)
    {
        auto out = Value(std::pow(val.get_data(), exp), {val.get_ptr(),});

        _Value<T>* val_ptr = val.get_ptr().get();
        _Value<T>* out_ptr = out.get_ptr().get();

        auto _back = [=]()
        {
            val_ptr->get_grad() += (exp * std::pow(val_ptr->get_data(), exp- static_cast<T>(1))) * out_ptr->get_grad();
        };
        out.set_backward(_back);

        return out;
    }

    template <class C>
    friend Value<C> operator+(C num, const Value<C>& val) {return val + num;}

    template <class C>
    friend Value<C> operator-(C num, const Value<C>& val) {return val - num;}

    template <class C>
    friend Value<C> operator*(C num, const Value<C>& val) {return val * num;}

    template <class C>
    friend Value<C> operator/(C num, const Value<C>& val) {return val / num;}

private:
    std::shared_ptr<_Value<T>> _ptr = nullptr;

    Value(const T& data, const std::vector<std::shared_ptr<_Value<T>>>& parents) { _ptr = std::make_shared<_Value<T>>(data, parents); }

public:
    // Constructors and destructors
    Value() { _ptr = std::make_shared<_Value<T>>(static_cast<T>(0)); }
    Value(const T& data) { _ptr = std::make_shared<_Value<T>>(data); }
    ~Value() { _ptr = nullptr; };

    // Copy and move constructors
    Value(const Value& other) { _ptr = other._ptr; }
    Value(Value&& other) { _ptr = other._ptr; other._ptr = nullptr; }

    // Copy and move assignment operators
    Value<T>& operator=(const Value<T>& other) { if (&other!=this) _ptr = other._ptr; return *this; }
    Value<T>& operator=(Value<T>&& other) { _ptr = other._ptr; other._ptr = nullptr; return *this; }

    // Transparency to the _Value class
    const T& get_data() const { return _ptr->get_data(); }
    const T& get_grad() const { return _ptr->get_grad(); }
    T& get_data() { return _ptr->get_data(); }
    T& get_grad() { return _ptr->get_grad(); }
    std::vector<std::shared_ptr<_Value<T>>>& get_parent_ptrs() const { return _ptr->get_parent_ptrs(); }
    void zero_grad() const { _ptr->zero_grad(); }
    void zero_grad_all() const { _ptr->zero_grad_all(); }
    void set_backward(std::function<void()> func) const { _ptr->set_backward(func); }
    void backward() const { _ptr->backward(); }
    void descend_grad(const T& learning_rate) const { _ptr->descend_grad(learning_rate); }
    std::vector<_Value<T>*> build_topo() const { return _ptr->build_topo(); }

    // ptr accessor
    std::shared_ptr<_Value<T>> get_ptr() const { return _ptr; }

    // Relu
    Value<T> relu() const
    {
        auto out = Value<T>(std::max(static_cast<T>(0), get_data()), {get_ptr(),});

        _Value<T>* this_ptr = get_ptr().get();
        _Value<T>* out_ptr = out.get_ptr().get();

        auto _back = [=]()
        {
            if (this_ptr->get_data() > static_cast<T>(0))
                this_ptr->get_grad() += out_ptr->get_grad();
        };
        out.set_backward(_back);

        return out;
    }

    // Arithmetic operators
    Value<T> operator+(const Value<T>& other) const
    {
        Value<T> out(
            get_data() + other.get_data(),
            {get_ptr(), other.get_ptr()}
        );

        _Value<T>* this_ptr = get_ptr().get();
        _Value<T>* other_ptr = other.get_ptr().get();
        _Value<T>* out_ptr = out.get_ptr().get();

        auto _back = [=]()
        {
            this_ptr->_grad += out_ptr->_grad;
            other_ptr->_grad += out_ptr->_grad;
        };
        out.set_backward(_back);

        return out;
    }

    Value<T> operator+(const T& other) const
    {
        auto temp = Value<T>(other);
        return operator+(temp);
    }

    Value<T> operator-(const Value<T>& other) const
    {
        auto out = Value<T>(
            get_data() - other.get_data(),
            {get_ptr(), other.get_ptr()}
        );
        
        _Value<T>* this_ptr = get_ptr().get();
        _Value<T>* other_ptr = other.get_ptr().get();
        _Value<T>* out_ptr = out.get_ptr().get();

        auto _back = [=]()
        {
            this_ptr->get_grad() += out_ptr->get_grad();
            other_ptr->get_grad() += out_ptr->get_grad();
        };
        out.set_backward(_back);

        return out;
    }

    Value<T> operator-(const T& other) const
    {
        auto temp = Value<T>(other);
        return operator-(temp);
    }

    Value<T> operator*(const Value<T>& other) const
    {
        auto out = Value<T>(
            get_data() * other.get_data(),
            {get_ptr(), other.get_ptr()}
        );

        _Value<T>* this_ptr = get_ptr().get();
        _Value<T>* other_ptr = other.get_ptr().get();
        _Value<T>* out_ptr = out.get_ptr().get();

        auto _back = [=]()
        {
            this_ptr->get_grad() += other_ptr->get_data() * out_ptr->get_grad();
            other_ptr->get_grad() += this_ptr->get_data() * out_ptr->get_grad();
        };
        out.set_backward(_back);

        return out;
    }

    Value<T> operator*(const T& other) const
    {
        auto temp = Value<T>(other);
        return operator*(temp);
    }

    Value<T> operator/(const Value<T>& other) const
    {
        auto temp = pow(other, static_cast<T>(-1));
        return operator*(temp);
    }

    Value<T> operator/(const T& other) const
    {
        auto temp = Value<T>(other);
        return operator/(temp);
    }

    Value<T> operator-()
    {
        return operator*(static_cast<T>(-1));
    }

    // Comparison operators
    bool operator==(const Value<T>& other) const
    {
        return get_data() == other.get_data();
    }

    bool operator<(const Value<T>& other) const
    {
        return get_data() < other.get_data();
    }

    bool operator>(const Value<T>& other) const
    {
        return get_data() > other.get_data();
    }

    bool operator<=(const Value<T>& other) const
    {
        return get_data() <= other.get_data();
    }

    bool operator>=(const Value<T>& other) const
    {
        return get_data() >= other.get_data();
    }
};

#endif