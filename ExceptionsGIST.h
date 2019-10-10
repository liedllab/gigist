#ifndef EXCEPTION_GIST_H
#define EXCEPTION_GIST_H

#include <exception>

class IndexOutOfRangeException : std::exception {
public:
  IndexOutOfRangeException() {}
};

class BoxInfoException : std::exception {
public:
  BoxInfoException() {}
};

#endif