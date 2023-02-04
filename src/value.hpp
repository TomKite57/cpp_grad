
#include<iostream>
#include<utility>
#include<functional>
#include<set>

enum operation {PLUS, MINUS, TIMES, DIVIDE};

//const std::function<void()> do_nothing = [](){return;};

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
    std::function<void()> _backward;


    Value(T data, const std::vector<Value<T>*> parents, operation op):
    _data{data}, _parents{parents}, _op{op}
    {}

public:
    Value(T data): _data{data} {}

    T get_data() const { return _data; }
    T get_grad() const { return _grad; }
    std::vector<Value<T>*> get_parent_ptrs() const { return _parents; }

    void zero_grad(){ _grad = 0; }

    void backward()
    {
        auto order = topological_sort(*this);

        // Set dx/dx=1
        _grad = 1.0;
        for (auto& n=order.rend(); n>=order.rbegin(); n++)
            n._backward();
    }

    Value<T> operator+(const Value<T>& other)
    {
        auto out = Value<T>(_data + other._data, {this, &other}, PLUS);
        return out;
    }
};

template<class T>
std::vector<Value<T>*> topological_sort(Value<T>& root)
{
    std::vector<Value<T>*> order;
    std::set<Value<T>*> visited;

    auto build_topo = [&](Value<T>& node)
    {
        if (visited.find(node) == visited.end())
        {
            visited.add(&node);
            for (auto& par : node.get_parent_ptrs())
                build_topo(par);
            order.push_back(&node);
        }
    };

    build_topo(root);

    return order;
}