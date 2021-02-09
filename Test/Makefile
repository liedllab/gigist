CXX = g++
CXXFLAGS =
GTEST_FLAGS = `pkg-config --cflags gtest_main`
GTEST_LIBS = `pkg-config --libs gtest_main`

.PHONY: tests all clean

tests: all
	./testapp

all: testapp

testapp: QuaternionTest.o LinkedCellGridTest.o main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(GTEST_LIBS)

QuaternionTest.o: QuaternionTest.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -c -o $@ $(GTEST_FLAGS) -DTESTS

LinkedCellGridTest.o: LinkedCellGridTest.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -c -o $@ $(GTEST_FLAGS) -DTESTS

main.o: MainTest.cpp
	$(CXX) $(CPPFlAGS) $(CXXFLAGS) $< -c -o $@ $(GTEST_FLAGS) -DTESTS

clean:
	rm testapp *.o
