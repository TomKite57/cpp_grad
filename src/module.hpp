

// Interface for NN components
class Module
{
public:
    Module() = delete;
    Module(const Module&) = delete;
    virtual ~Module() = default;

    virtual std::vector<std::shared_ptr<Module>> get_parameters() = 0;

    // [TODO] Value must inherit from here?
    virtual void zero_grad()
    {
        for (auto& node : get_parameters)
            node->zero_grad();
    }

    // Need different arguments for each derived class
    virtual void operator()() = 0;
};

template <class T>
class Neuron: public Module
{
private:
    size_t _size;
    std::vector<Value<T>> _weights;
    Value<T> _bias;

public:
    Neuron(size_t size): Module{}, _size{size}
    {
        _weights = std::vector<Value<T>>(size, static_cast<T>(0));
        _bias = Value<T>(static_cast<T>(0));
    }
    Neuron(const Neuron&) = delete;
    virtual ~Neuron() = default;
};

//class Layer: public Module
//{
//
//};
//
//class MLP: public Module
//{
//
//};
