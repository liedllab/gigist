#include "../LinkedCellGrid.h"
#include <gtest/gtest.h>


TEST( LinkedCellGrid, OuterIteratorTest )
{
    LinkedCellGrid<int> test1;
    int cnt{ 0 };
    for( auto it : test1 ) {
        cnt++;
    }
    EXPECT_EQ(cnt, 0);
}

TEST(LinkedCellGrid, ConstructorTest)
{
    LinkedCellGrid<int> grid{ 20, 200 };
    EXPECT_EQ( grid.getSize(), 20 );
    EXPECT_EQ( grid.getTotalDataSize(), 0 );
    for (auto i : grid ) {
        EXPECT_EQ(i.get(), -1);
        int cnt{ 0 };
        for (auto j : i) {
            cnt++;
        }
        ASSERT_EQ(cnt, 0);
    }
}



TEST(LinkedCellGrid, PushBackTest)
{
    LinkedCellGrid<int> grid{ 20, 200 };
    grid.push_back(0, 1);
    grid.push_back(0, 2);
    grid.push_back(1, 4);
    grid.push_back(2, 3);
    grid.push_back(2, 4);
    grid.push_back(1, 2);
    std::vector<int> results;
    std::vector<int> results2;
    for (LinkedCellGrid<int>::OuterIterator i : grid)
    {
        for (auto j : i)
        {
            if (i.getIndex() == 0) {
                results.push_back( j );
            } else {
                results2.push_back( j );
            }
        }
    }
    ASSERT_EQ( results.size(), 2 );
    EXPECT_EQ( results.at(0), 1 );
    EXPECT_EQ( results.at(1), 2 );
    EXPECT_EQ( results2.size(), 4 );
    EXPECT_EQ( results2.at(0), 4 );
    EXPECT_EQ( results2.at(1), 2 );
    EXPECT_EQ( results2.at(2), 3 );
    EXPECT_EQ( results2.at(3), 4 );
}