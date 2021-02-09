#include <algorithm>
#include <vector>
#include <stdexcept>

template<class T>
class LinkedCellGrid{
private:
    // class Node{
    // private:
    //     int m_next = -1;
    //     T m_data;
    // public:
    //     Node() = default;
    //     Node( const Node& ) = default;
    //     Node( Node&& ) = default;
    //     Node& operator=( const Node& ) = default;
    //     Node& operator=( Node&& ) = default;
    //     ~Node() = default;

    //     const T& at(int idx) const
    //     {
    //         if (idx <= 0) {
    //             return m_data;
    //         }
    //         return 
    //     }
    // };
    std::vector<std::pair<int, T>> m_data;
    std::vector<int> m_startIndices;
    std::vector<int> m_endIndices;
public:

    

    class InnerIterator{
    
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<int, T>;
        using pointer = std::pair<int, T>*;
        using reference = std::pair<int, T>&;


        InnerIterator(int node, LinkedCellGrid<T>* parent)
        : m_current{ node }
        , m_parent{ parent }
        {}


        reference operator*() const { return m_parent->m_data.at(m_current); }
        pointer operator->() { return &m_parent->m_data.at(m_current); }

        InnerIterator& operator++() 
        { 
            m_current = std::get<0>(m_parent->m_data.at(m_current));
            return *this;
        }
        InnerIterator operator++(int) { InnerIterator tmp = *this; ++(*this); return tmp; }
        
        friend bool operator==( const InnerIterator& lhs, const InnerIterator& rhs )
        {
            return lhs.m_current == rhs.m_current;
        }
        friend bool operator!=( const InnerIterator& lhs, const InnerIterator& rhs )
        {
            return lhs.m_current != rhs.m_current;
        }
    private:
        int m_current;
        LinkedCellGrid<T>* m_parent;

    };

    class OuterIterator{
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = int;
        using pointer           = int*;
        using reference         = int&;
        OuterIterator( pointer ptr, LinkedCellGrid<T>& parent )
        : m_ptr{ ptr }
        , m_parent{ &parent }
        {}

        const OuterIterator& operator*() const { return *this; }
        OuterIterator* operator->() { return this; }

        const int& get() const { return *m_ptr; }
        int getIndex() const { return m_gridIndex; }

        OuterIterator& operator++() { m_ptr++; m_gridIndex++; return *this; }
        OuterIterator operator++(int) { OuterIterator tmp = *this; ++(*this); return tmp; }
        
        friend bool operator==( const OuterIterator& lhs, const OuterIterator& rhs )
        {
            return lhs.m_ptr == rhs.m_ptr;
        }
        friend bool operator!=( const OuterIterator& lhs, const OuterIterator& rhs )
        {
            return lhs.m_ptr != rhs.m_ptr;
        }

        const InnerIterator begin() const { return InnerIterator(*m_ptr, m_parent); }
        const InnerIterator end() const { return InnerIterator(-1, m_parent); }
        InnerIterator begin() { return InnerIterator(*m_ptr, m_parent); }
        InnerIterator end() { return InnerIterator(-1, m_parent); }
    
    private:
        pointer m_ptr;
        int m_gridIndex = 0;
        LinkedCellGrid<T>* m_parent;
    };

    // Rule of 0
    LinkedCellGrid() = default;
    LinkedCellGrid(const LinkedCellGrid&) = default;
    LinkedCellGrid(LinkedCellGrid&&) = default;
    LinkedCellGrid& operator=(const LinkedCellGrid&) = default;
    LinkedCellGrid& operator=(LinkedCellGrid&&) = default;
    ~LinkedCellGrid() = default;

    explicit LinkedCellGrid(int size, int numberDataPoints)
    : m_startIndices( size, -1 )
    , m_endIndices( size, -1 )
    {
        m_data.reserve(numberDataPoints);
    }

    OuterIterator       begin()        { return OuterIterator( &m_startIndices[0], *this);  }
    const OuterIterator begin()  const { return OuterIterator( &m_startIndices[0], *this);  }
    const OuterIterator cbegin() const { return OuterIterator( &m_startIndices[0], *this); }
    OuterIterator       end()          { return OuterIterator( &m_startIndices[m_startIndices.size()], *this);    }
    const OuterIterator end()    const { return OuterIterator( &m_startIndices[m_startIndices.size()], *this);    }
    const OuterIterator cend()   const { return OuterIterator( &m_startIndices[m_startIndices.size()], *this);   }


    void resize(int size, int numberDataPoints)
    {
        m_data.reserve(numberDataPoints);
        m_startIndices.resize(size, -1);
        m_endIndices.resize(size, -1);
    }

    const T& at(int gridIdx, int idx) const
    {
        int next = m_startIndices.at(gridIdx);
        while( idx > 0 ) {
            next = std::get<0>(m_data.at(next));
            if (next == -1)
                throw std::out_of_range("Inner Index is out of range, consider using iterators.");
            idx--;
        }
        return std::get<1>(m_data.at(next));
    }

    size_t getSize() const
    {
        return m_startIndices.size();
    }

    size_t getTotalDataSize() const
    {
        return m_data.size();
    }

    void push_back(int idx, T value)
    {
        int pos = m_data.size();
        if (m_startIndices.at(idx) == -1) {
            m_startIndices.at(idx) = pos;
            m_endIndices.at(idx) = pos;
            m_data.push_back({-1, value});
        } else {
            m_data.at(m_endIndices.at(idx)).first = pos;
            m_endIndices.at(idx) = pos;
            m_data.push_back({-1, value});
        }
    }


};