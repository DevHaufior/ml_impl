#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "../util/maths.h"
using std::rand;
using std::srand;
using std::time;

// using std::vector;
using std::cout;
using std::endl;

using maths::matrix::dot;
using maths::function::sigmoid;

class LogisticRegression {
    private:
        double learningRate;
        int numIter;
        vector<double> weights;

    public:
    LogisticRegression(double learningRate=0.01, int numIter=100): learningRate(learningRate), numIter(numIter) {

    }
    vector<double> getWeights() {
        return this->weights;
    }
    void fit(vector<vector<double> > &dataMatrix, vector<int> &classLables) {
        for (auto &row: dataMatrix) {
            row.push_back(1.0);
        }
        weights.resize(dataMatrix.front().size());
        random_init(weights);
        gradientDescentAndUpdateWeights(dataMatrix, classLables);
    }

    void random_init(vector<double> &x) {
        srand((int) time(0));
        for (int i = 0; i < x.size(); i++) {
            x[i] = ((rand() % 10 ) + 1) / 10;
        }
    }

    void gradientDescentAndUpdateWeights(vector<vector<double> > &dataMatrix, vector<int> &classLables) {
        for (int i = 0; i < numIter; i++) {
            vector<double> grands(weights.size(), 0.0);
            // do grands
            for (int i = 0; i < dataMatrix.size(); i++) {
                double error = sigmoid(dot(weights, dataMatrix[i])) - classLables[i];
                for (int j = 0; j < dataMatrix.front().size(); j++) {
                    grands[j] += error * dataMatrix[i][j];
                }
            }
            //update weights
            for (int i = 0; i < weights.size(); i++) {
                weights[i] -= learningRate * grands[i];
            }
        }
    }

    int predict(vector<double> &x) {
        return dot(weights, x) >0 ? 1: 0;
    }
    
    double predictprob(vector<double> &x) {
        return sigmoid(dot(weights, x));
    }
};


int main() {
    vector<vector<double> > dataMatrix{
            {0, 1.0},
            {1.0, 0},
            {3.0,1.0},
            {1.0, 3.0},
            {1, 3}
        };
    vector<int> classLables{
            1,
            0,
            0,
            1,
            1
    };

    LogisticRegression lr{0.01,100};
    lr.fit(dataMatrix, classLables);
    for (auto &x: dataMatrix) {
        cout<< lr.predict(x) <<endl;
    }
    for (auto &w: lr.getWeights()) {
        cout<<w<<" a"<<endl;;
    }
    return 0;
}


