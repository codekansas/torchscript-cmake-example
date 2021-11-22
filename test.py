import torch
from torch import Tensor, nn
from torch.jit._script import script

import libcpp


class TestModule(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x + 1


def main() -> None:
    tensor = torch.randn(3, 4)
    module = script(TestModule())

    ident_tensor = libcpp.tensor_identity(tensor)  # This passes
    ident_module = libcpp.module_identity(module)  # This fails


if __name__ == "__main__":
    main()
