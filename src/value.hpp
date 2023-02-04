
#include<iostream>
#include<utility>
#include<functional>
#include<set>

enum operation {PLUS, MINUS, TIMES, DIVIDE};

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

private:
    T _data{0};
    T _grad{0};
    std::vector<Value<T>*> _parents;
    operation _op;
    std::function<void()> _backward = do_nothing;


    Value(T data, std::vector<Value<T>*> parents, operation op):
    _data{data}, _parents{parents}, _op{op}
    {}

public:
    Value(T data): _data{data} {}

    T get_data() const { return _data; }
    T get_grad() const { return _grad; }
    std::vector<Value<T>*> get_parent_ptrs() { return _parents; }

    void zero_grad(){ _grad = 0; }

    void backward()
    {
        //auto order = topological_sort(*this);
        auto order = build_topo(this);

        // Set dx/dx=1
        _grad = 1.0;
        for (auto n=order.rbegin(); n!=order.rend(); ++n)
        {
            (*n)->_backward();
        }
    }

    Value<T> operator+(Value<T>& other)
    {
        auto out = Value<T>(_data + other._data, {this, &other}, PLUS);

        auto _back = [&]()
        {
            _grad += out._grad;
            other._grad += out._grad;
        };
        out._backward = _back;

        return out;
    }

    Value<T> operator*(Value<T>& other)
    {
        auto out = Value<T>(_data * other._data, {this, &other}, TIMES);

        auto _back = [&]()
        {
            _grad += other._data * out._grad;
            other._grad += _data * out._grad;
        };
        out._backward = _back;

        return out;
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

// Make class for encapsulation
template<class T>
void _build_topo(Value<T>* node, std::set<Value<T>*>& visited, std::vector<Value<T>*>& order)
{
    if (visited.find(node) != visited.end())
        return;

    visited.insert(node);
    for (auto par_ptr : node->get_parent_ptrs())
        _build_topo(par_ptr, visited, order);
    order.push_back(node);
}
