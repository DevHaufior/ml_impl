#include <iostream>
#include <Eigen/Dense>
#include <cassert>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <ctime>

using std::rand;
using std::srand;
using std::time;

using namespace std;
using namespace Eigen;

class FM {
public:
    FM(int k, int num_iters=1000, double learning_rate=0.01):
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


        srand((int) time(0));
        this->_w0 = ((rand() % 10 ) + 1) / 10;
        this->_weights = VectorXd::Random(X_train.front().size());
        // this->_vMatrix.resize(X_train.front().size());
        // for (int i = 0; i < this->_vMatrix.size(); i++) {
        //     this->_vMatrix[i] = VectorXd::Random(this->_k);
        // }
        this->_vMatrix = MatrixXd::Random(this->_k, X_train.front().size());
        // this->init_parameters();
        
        for (int i = 0; i < this->_num_iters; i++) {

            // weight = weight - _learning_rate * dloss/dweight
            gradientDescend(X_train, y_train);
        }
    }
    
    double calcuteOne(vector<double> &x, VectorXd &xVector, vector<double> &vF) {
 
        vF.resize(this->_k);
        xVector.resize(x.size());

        double result = this->_w0;

        for (int i = 0; i < x.size(); i++){
            xVector[i] = x[i];
        }
        result += this->_weights.dot(xVector);

        double cross_f_result = 0;

        for (int i = 0; i < this->_k; i++) {

            vF[i] = this->_vMatrix.row(i).dot(xVector);
            cross_f_result += vF[i] * vF[i] - xVector.cwiseProduct(xVector).dot((this->_vMatrix.row(i).cwiseProduct(this->_vMatrix.row(i))));
        }

        result += 0.5 * cross_f_result;
        return result;
    }

    void gradientDescend(vector<vector<double>> &X_train, vector<double> &y_train) {

        double total_grad_w0 = 0;
        VectorXd total_grad_weights = VectorXd::Zero(X_train.front().size());
        
        MatrixXd total_grad_vMatrix = MatrixXd::Zero(this->_k, X_train.front().size());

        for (int n = 0; n < X_train.size(); n++) {
            vector<double> vF;
            VectorXd xVector;
            double loss = y_train[n] -  calcuteOne(X_train[n], xVector, vF);

            total_grad_w0 += loss * 1;
            total_grad_weights = total_grad_weights + loss * xVector;

            for (int i = 0; i < this->_k; i++) {
                total_grad_vMatrix.row(i) += loss * (vF[i] * xVector.transpose() - this->_vMatrix.row(i).cwiseProduct((xVector.cwiseProduct(xVector)).transpose()));
            }
        }
        this->_w0 = this->_w0 + this->_learning_rate * total_grad_w0;
        this->_weights = this->_weights + this->_learning_rate * total_grad_weights;
        this->_vMatrix = this->_vMatrix + this->_learning_rate * total_grad_vMatrix;
    }


    vector<double> predict(vector<vector<double>> &X) {
        vector<double> result;
        for (auto &x: X) {
            vector<double> vF;
            VectorXd xVector;
            result.push_back(calcuteOne(x, xVector, vF));
        }
        return result;
    }
public:
    string object;// 默认是回归任务
    int _k;
    int _num_iters; // 迭代次数
    double _learning_rate;

    double _w0;
    VectorXd _weights; // <w_0, w_1, ..., w_n>
    MatrixXd _vMatrix; // correspond to x factors 
};

int main() {    
    FM fm{5};
    vector<vector<double>> X_train{{1,2}, {2,4}}; 
    vector<double> y_train{1, 2};
    // vector<vector<double>> X(2, {1,2});
    fm.fit(X_train, y_train);
    cout<<"w0="<<fm._w0<<endl;
    cout<<"weights="<< fm._weights<<endl;
    cout<<"vmatrix="<<fm._vMatrix<<endl;

    cout<<"predict="<< (X_train).size() << endl;
    for(auto & rt: fm.predict(X_train)) {
        cout<<rt<<" ";
    }
    cout<<endl;
}
// g++ -I /Users/hufei/Documents/StudyMaterials/C++/C++_UTILS/Eigen/eigen-eigen main.cpp