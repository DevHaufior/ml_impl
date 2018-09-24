#include "ps/ps.h"
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <utility>
// #include <math>

using namespace ps;
using namespace std;

using std::rand;
using std::srand;
using std::time;


/**
 * \brief an example handle adding pushed kv into store
 */
template <typename Val>
struct GradientDescentHandler
{

  public:
    GradientDescentHandler(double learning_rate = 0.01) : _learning_rate(learning_rate) {
    }

    void operator()(const KVMeta &req_meta, const KVPairs<Val> &req_data, KVServer<Val> *server) {
        if (!this->is_inited) {

            this->init_params(req_data);
            this->is_inited = true;
        }

        size_t n = req_data.keys.size();
        KVPairs<Val> res;
        if (req_meta.push) {
            CHECK_EQ(n, req_data.vals.size());
        }
        else {
            res.keys = req_data.keys;
            res.vals.resize(n);
        }
        for (size_t i = 0; i < n; ++i) {
            Key key = req_data.keys[i];
            if (req_meta.push){
                store[key] += this->_learning_rate * req_data.vals[i];
            }
            else {
                res.vals[i] = store[key];
            }
        }

        server->Response(req_meta, res);
    }

  private:
    void init_params(const KVPairs<Val> &req_data) {
        srand((int) time(0));
        for (size_t i = 0; i < req_data.keys.size(); i++) {
            store[req_data.keys[i]] = (rand() % 10 + 1) / 10.0;
        }
        std::cout << "initialise fm model parameters" << std::endl;
    }

    std::unordered_map<Key, Val> store; // 存储Key-Value的值

    double _learning_rate;
    bool is_inited = false;
};

void StartServer()
{
    if (!IsServer())
    {
        return;
    }
    std::cout << "start server" << std::endl;
    auto server = new KVServer<float>(0);
    auto gd_handler = GradientDescentHandler<float>(0.01);
    server->set_request_handle(gd_handler);
    RegisterExitCallback([server]() { delete server; });
}


// float calcuteOne(vector<float> &x, vector<Key> &keys, vector<float> &vals, vector<float> &vF) {
//     float result = vals[0];

//     // result += this->_weights.dot(xVector);
//     for (int i = 0; i < x.size(); i++) {
//         result += x[i] * vals[i+1];
//     }

//     float cross_f_result = 0;
//     for (int i = 0; i < vF.size(); i++) {
//         vF[i] = 0;
//         float squreVF = 0;
//         for (int j = 0; j < x.size(); j++) {
//             float temp = x[j] * vals[1 + x.size() + i*x.size()+j];
//             vF[i] += temp;
//             squreVF +=  temp * temp;
//         }
//         cross_f_result += vF[i] * vF[i] - squreVF;
//     }
//     result += 0.5 * cross_f_result;
//     return result;
// }

vector<vector<float>> X_train{{1, 2}, {2, 4}, {2,1}};
vector<int> y_train{1, 2,1};

std::pair<vector<vector<float>>, vector<int>> get_training_data(int rank) {
    int num_workers = NumWorkers();
    vector<vector<float>> x;
    vector<int> y;
    for (size_t i = 0; i * num_workers + rank < X_train.size(); i++) {
        x.push_back(X_train[i * num_workers + rank]);
        y.push_back(y_train[i * num_workers + rank]);
    }
    return make_pair(x, y);
}

template<typename T>
void print_vector(vector<T> &data) {
    for (auto &t: data) {
        std::cout<<t<< ",";
    }
    std::cout<<std::endl;
}

float calculate_predicted_value(vector<Key> &keys, vector<float> &vals, vector<float> &x, vector<float> &f_all_x, int k, bool is_predict=true) {
    float res = 0;
    res += vals[0]; // w_0
    for (size_t t = 0; t < x.size(); t++) {
        res += x[t] * vals[1 + t]; // w_i * x_i
    }

    float cross_value = 0;
    for (int f = 0; f < k; f++) {
        float v_x_sum = 0;
        float v_x_square_sum = 0;

        for (size_t i = 0; i < x.size(); i++) {
            float v_if = vals[1 + x.size() + f * x.size() + i];
            v_x_sum += v_if * x[i];
            v_x_square_sum += std::pow(v_if * x[i], 2);
        }

        cross_value += (std::pow(v_x_sum, 2) - v_x_square_sum);
        if (!is_predict) {
            f_all_x[f] = v_x_sum;
        } 
    }

    res += 0.5 * cross_value;
    return res;
}


void calculate_grad(vector<Key> &keys, vector<float> &vals, vector<vector<float>> &X_train, vector<int> &y_train,vector<float> &grads, int k) {
    
    vector<float> f_all_x(k);

    for (size_t m = 0; m < X_train.size(); m++) {
        auto &x = X_train[m];

        for(size_t t = 0; t < f_all_x.size(); t++) {
            f_all_x[t] = 0;
        }

        float y_pred = calculate_predicted_value(keys, vals, x, f_all_x, k, false);
        float loss = y_train[m] - y_pred;

        // grads += &y_pred / &w;
        grads[0] += loss * 1; // w_0;

        for (size_t i = 0; i < x.size(); i++) {
            grads[i+1] += loss * x[i]; // w_1~n
        }

        for (size_t i = 0; i < x.size(); i++) {
            for (int f = 0; f < k; f++) {
                float v_if = vals[1 + x.size() + f * x.size() + i];
                grads[1 + x.size() + f * x.size() + i] += loss * (x[i] * f_all_x[f] - v_if * std::pow(x[i], 2));
                // v_if
            }
        }
    }

}

void RunWorker()
{
    if (!IsWorker())
        return;
    KVWorker<float> kv(0, 0);
    int rank = MyRank();
    std::cout << "start worker rank[" << rank << "]" << std::endl;
    auto data = get_training_data(rank);
    auto X_train = data.first;
    auto y_train = data.second;

    int num_iter = 800;
    int k = 6;

    int num_params = 1 + X_train.front().size() + X_train.front().size() * k;
    vector<Key> keys(num_params);
    vector<float> vals(num_params);

    for (size_t t = 0; t < keys.size(); t++) {
        keys[t] = t;
    }

    vector<float> grads(keys.size());
    for(int num = 0; num < num_iter; num++) {
        kv.Wait(kv.Pull(keys, &vals));
        // if (num == 0) {
        //             print_vector<Key>(keys);
        // print_vector<float>(vals);
        // }

        for (size_t t = 0; t < grads.size(); t++) {
            grads[t] = 0;
        }
        calculate_grad(keys, vals, X_train, y_train, grads, k);

        kv.Wait(kv.Push(keys, grads));

        // if (num == 1) {
        //     break;
        // }

    }

    if (rank==0) {
        std::cout<<"learned paramters:"<<std::endl;
        for (size_t m = 0; m < keys.size(); m++) {
            std::cout<<"("<<keys[m]<<", "<<vals[m]<<") "<<std::endl;
        }
    }

    vector<float> tmp;
    for (size_t m = 0; m < X_train.size(); m++) {
        auto &x = X_train[m];
        std::cout<<"y_real="<<y_train[m]<<", y_pred="<<calculate_predicted_value(keys, vals, x, tmp, k)<<std::endl;
    }
}

int main(int argc, char *argv[])
{
    // start system
    Start(0);
    // setup server nodes
    StartServer();
    // run worker nodes
    RunWorker();
    // stop system
    Finalize(0, true);
    return 0;
}
// g++ -std=c++11 -msse2 -fPIC -O3 -ggdb -Wall -finline-functions  -I../include -o disFM disFM.cc -L../lib -lprotobuf-lite -lzmq -lps -pthread