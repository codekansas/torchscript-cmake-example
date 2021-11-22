#include "example.h"

namespace py = pybind11;

torch::jit::Module module_identity(const torch::jit::Module &mod) {
  return mod;
}
torch::Tensor tensor_identity(const torch::Tensor &tnsr) { return tnsr; }

PYBIND11_MODULE(MODULE_NAME, m) {
  m.def("module_identity", &module_identity);
  m.def("tensor_identity", &tensor_identity);
}
