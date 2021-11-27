import torch
import torch.utils.cpp_extension
from torch import Tensor, nn

import cpp


class TestModule(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x + 1


def main() -> None:
    tensor = torch.randn(3, 4)
    module = torch.jit.script(TestModule())

    ident_tensor = cpp.tensor_identity(tensor)  # This passes
    ident_module = cpp.module_identity(module._c)  # This fails

    print("ident tensor:", ident_tensor)
    print("ident module:", ident_module)


if __name__ == "__main__":
    main()
