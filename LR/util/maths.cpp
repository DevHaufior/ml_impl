#include"./maths.h"
namespace maths {
    double function::sigmoid(double x) {
        return 1.0 / (1 + exp(-1 * x));
    }

    double matrix::dot(vector<double> x, vector<double> y) {
        double result;
        for (int i = 0; i < x.size(); i++) {
            result += x[i] * y[i];
        }
        return result;
    }
}
