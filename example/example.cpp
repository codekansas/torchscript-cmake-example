#define EXPORT __attribute__((visibility("default")))

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/script.h>

EXPORT torch::jit::Module module_identity(const torch::jit::Module &mod) {
  return mod;
}
EXPORT torch::Tensor tensor_identity(const torch::Tensor &tnsr) { return tnsr; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("module_identity", &module_identity);
  m.def("tensor_identity", &tensor_identity);
}
