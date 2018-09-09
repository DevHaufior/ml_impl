#ifndef MATHS_H
#define MATHS_H

#include<vector>
#include <cmath>
using std::vector;

namespace maths {

    namespace matrix {
        double dot(vector<double> x, vector<double> y);
    }

    namespace function {
        double sigmoid(double x);
    }
}

#endif  // MATHS_H