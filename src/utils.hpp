
#include<iostream>
#include<vector>
#include<cstdlib>

// Vector printout
template <typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T>& v)
{
    os << "Vector(";
    for (auto ptr=v.begin(); ptr!=v.end(); ++ptr)
    {
        os << *ptr;
        if (ptr != v.end()-1)
            os << ", ";
    }
    os << ")";
    return os;
}

void set_seed()
{
    srand((unsigned) time(NULL));
}