"""Verify whether detect_anomaly causes N extra forward passes."""
import torch

torch.autograd.set_detect_anomaly(True, check_nan=False)

x = torch.randn(2, 3, requires_grad=True)
y = torch.randn(2, 3, requires_grad=True)
z = (x * y).sum()
print(f"Before backward: x._version={x._version}, y._version={y._version}")

# Call backward with detect_anomaly; if it does an extra forward, versions change
z.backward()
print(f"After backward: x._version={x._version}, y._version={y._version}")
print(f"x.grad: {x.grad}")

# Test 2: if we do backprop twice, does the second one detect "already allocated"?
z2 = (x * y).sum()
print(f"\nTest 2: before 2nd backward: x._version={x._version}, y._version={y._version}")
z2.backward()
print(f"After 2nd backward: x._version={x._version}, y._version={y._version}")

torch.autograd.set_detect_anomaly(False)
