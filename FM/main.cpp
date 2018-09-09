#include <iostream>
#include <Eigen/Dense>
#include <cassert>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;
using namespace Eigen;

class FM {
public:
    FM(int k, int num_iters=100):
    object("regression"), 
    _num_iters(num_iters),
    _learning_rate(learning_rate), 
    _k(k) {
        
    }

    void init_parameters() {

    }

    void fit(vector<vector<double>> &X_train, vector<double> &y_train) {
        assert(X_train.size() == y_train.size() && X_train.size() != 0);
        assert(X_train.front().size() !=0);
        auto min_max = std::minmax_element(X_train.cbegin(), X_train.cend(), [](const vector<double> &a, const vector<double> &b) -> bool {
             return a.size() < b.size();
         });
        assert(min_max.first->size() == min_max.second->size());

        this->init_parameters();

        for (int i = 0; i < this->_num_iters; i++) {

            // weight = weight - _learning_rate * dloss/dweight

        }

    }
    
    VectorXd predict(vector<vector<double>> &X) {
        return VectorXd::Random(4);
    }
private:
    string object;// 默认是回归任务
    int _k;
    VectorXd weights; // <w_0, w_1, ..., w_n>
    MatrixXd v_matrix; // correspond to x factors 
    int _num_iters; // 迭代次数
    double _learning_rate;
};

int main() {
    FM fm{5};
    vector<vector<double>> X_train{{1,2}, {1,2}}; 
    vector<double> y_train{1, 3};
    vector<vector<double>> X(2, {1,2});
    fm.fit(X_train, y_train);
    cout<< fm.predict(X);
}