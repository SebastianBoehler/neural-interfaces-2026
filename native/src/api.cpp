#include "ni26/api.h"
#include "ni26/runtime.h"
#include "ni26/math.h"
#include <torch/torch.h>
#include <memory>
#include <sstream>

namespace {
thread_local std::string error;
template<class F> int checked(F operation) {
  try { error.clear();operation();return 0; }
  catch(const std::exception& exception) { error=exception.what();return -1; }
  catch(...) { error="unknown native runtime failure";return -1; }
}
ni26::Runtime& runtime(void* model) {
  ni26::require(model!=nullptr,"null native model");
  return *static_cast<ni26::Runtime*>(model);
}
}
extern "C" const char* ni26_error() { return error.c_str(); }
extern "C" void* ni26_create(const char* directory,const char* device,const char* channels) {
  std::unique_ptr<ni26::Runtime> model;
  int status=checked([&] {
    ni26::require(directory && device && channels,"null native model metadata");
    auto root=std::filesystem::path(directory);
    torch::set_num_threads(1);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    auto params=ni26::Params::load(root/"native.pt");
    model=std::make_unique<ni26::Runtime>(params,torch::Device(device),root);
    std::vector<std::string> names;
    std::stringstream stream(channels);std::string name;
    while(std::getline(stream,name,',')) names.push_back(name);
    model->set_channels(names);
  });
  return status==0?model.release():nullptr;
}
extern "C" void* ni26_clone(void* source) {
  std::unique_ptr<ni26::Runtime> copy;
  int status=checked([&] { copy=std::make_unique<ni26::Runtime>(runtime(source)); });
  return status==0?copy.release():nullptr;
}
extern "C" void ni26_destroy(void* model) { delete static_cast<ni26::Runtime*>(model); }
extern "C" int ni26_reset(void* model) { return checked([&] { runtime(model).reset(); }); }
extern "C" int ni26_predict(void* model,const float* input,int64_t count,int cuda,int64_t* labels) {
  return checked([&] {
    ni26::require(input && labels && count>0 && (cuda==0 || cuda==1),"invalid native prediction buffer");
    auto options=torch::TensorOptions().dtype(torch::kFloat32).device(cuda?torch::kCUDA:torch::kCPU);
    auto x=torch::from_blob(const_cast<float*>(input),{count,41,480},options);
    auto result=runtime(model).predict(x).to(torch::kCPU).contiguous();
    std::copy(result.data_ptr<int64_t>(),result.data_ptr<int64_t>()+count,labels);
  });
}
