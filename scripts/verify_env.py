"""setup-venv.ps1後のGPU認識を確認する（AC3）。D-2の順序が壊れるとCPU版へ落ちる。"""
import torch

print(f"torch.__version__ = {torch.__version__}")
print(f"torch.version.cuda = {torch.version.cuda}")

if not torch.cuda.is_available():
    print("torch.cuda.is_available() = False -> CPU版（D-2の順序が壊れている可能性）")
    raise SystemExit(1)

capability = torch.cuda.get_device_capability(0)
arch_list = torch.cuda.get_arch_list()
print(f"torch.cuda.get_device_capability(0) = {capability}")
print(f"torch.cuda.get_arch_list() = {arch_list}")
print(f"sm_120 in arch_list = {'sm_120' in arch_list}")

x = torch.randn(2048, 2048, device="cuda")
y = torch.randn(2048, 2048, device="cuda")
z = x @ y
torch.cuda.synchronize()
print(f"GPU matmul smoke test: shape={tuple(z.shape)} device={z.device}")
