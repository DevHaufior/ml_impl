#include "ps/ps.h"
#include <cmath>
using namespace ps;

void StartServer() {
  if (!IsServer()) {
    return;
  }
  std::cout<<"start server"<<std::endl;
  auto server = new KVServer<float>(0);
  server->set_request_handle(KVServerDefaultHandle<float>());
  RegisterExitCallback([server](){ delete server; });
}

void RunWorker() {
  if (!IsWorker()) return;
  KVWorker<float> kv(0, 0);

  // init
  std::cout<<"start worker"<<std::endl;
  int num = 10000;
  std::vector<Key> keys(num);
  std::vector<float> vals(num);

  int rank = MyRank();
  srand(rank + 7);
  for (int i = 0; i < num; ++i) {
    keys[i] = kMaxKey / num * i + rank;
    vals[i] = (rand() % 1000);
  }

  // push
  int repeat = 50;
  std::vector<int> ts;
  for (int i = 0; i < repeat; ++i) {
    ts.push_back(kv.Push(keys, vals));
    std::cout<<"here is i="<<i<<std::endl;
    // to avoid too frequency push, which leads huge memory usage
    if (i > 10) kv.Wait(ts[ts.size()-10]);
  }
  for (int t : ts) kv.Wait(t);

  // pull
  std::vector<float> rets;
  kv.Wait(kv.Pull(keys, &rets));

  float res = 0;
  for (int i = 0; i < num; ++i) {
    res += std::fabs(rets[i] - vals[i] * repeat);
  }

  CHECK_LT(res / repeat, 1e-5);
  LL << "error: " << res / repeat;
}

int main(int argc, char *argv[]) {
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
// g++ -std=c++0x -std=c++11 -msse2 -fPIC -O3 -ggdb -Wall -finline-functions -I./src -I./include -I/Users/hufei/Downloads/temp/ps-lite/deps/include  -o tests/test_simple_app tests/test_simple_app.cc build/libps.a -Wl,-rpath,/Users/hufei/Downloads/temp/ps-lite/deps/lib -L/Users/hufei/Downloads/temp/ps-lite/deps/lib -lprotobuf-lite -lzmq -pthread

// g++ -std=c++0x -std=c++11 -msse2 -fPIC -O3 -ggdb -Wall -finline-functions -I./src -I./include -o test_kv_app test_kv_app.cc lib/libps.a -Wl,-rpath,lib -L./lib -lprotobuf-lite -lzmq -pthread

//// g++ -std=c++0x -std=c++11 -msse2 -fPIC -O3 -ggdb -Wall -finline-functions  -I./include -o test_kv_app test_kv_app.cc lib/libps.a -Wl,-rpath,lib -L./lib -lprotobuf-lite -lzmq -pthread
// ./local.sh 1 2 ./test_kv_app