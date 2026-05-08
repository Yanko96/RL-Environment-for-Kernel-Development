import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main():
    print(f"cuda: {torch.cuda.is_available()}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"triton: {triton.__version__}")

    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    _add_kernel[(4,)](x, y, out, n, BLOCK_SIZE=256)
    print(f"triton kernel ok: {torch.allclose(out, x + y)}")


if __name__ == "__main__":
    main()
