"""Test Storage._version behavior in PyTorch 2.12."""
import torch, ctypes, sys

print(f"PyTorch {torch.__version__}")

buf = torch.zeros(3, requires_grad=False)
print(f"buf._version = {buf._version}")

clone_detach = buf.clone().detach()
print(f"clone_detach._version = {clone_detach._version}")
print(f"Same storage: {buf.data_ptr() == clone_detach.data_ptr()}")

# Now copy to buffer using plain copy_
buf.copy_(torch.ones(3))
print(f"\nAfter buf.copy_():")
print(f"  buf._version = {buf._version}")
print(f"  clone_detach._version = {clone_detach._version}")

# Test Storage version
print(f"\nBuffer untyped_storage:")
print(f"  buf.untyped_storage() = {buf.untyped_storage()}")
print(f"  buf.storage_version() = {buf.untyped_storage._version if hasattr(buf.untyped_storage, '_version') else 'N/A'}")
print(f"  clone_detach.untyped_storage() version = {clone_detach.untyped_storage._version if hasattr(clone_detach.untyped_storage, '_version') else 'N/A'}")

# Test 2: detach only
buf2 = torch.zeros(3, requires_grad=False)
detached = buf2.detach()
print(f"\nTest2: buf2.detach()")
print(f"  buf2._version={buf2._version}, detached._version={detached._version}")
print(f"  Same storage: {buf2.data_ptr() == detached.data_ptr()}")
buf2.copy_(torch.ones(3))
print(f"After buf2.copy_(): buf2._version={buf2._version}, detached._version={detached._version}")

# Test 3: .data.copy_() 
buf3 = torch.zeros(3, requires_grad=False)
data_copy = buf3.detach()
buf3.data.copy_(torch.ones(3))
print(f"\nTest3: buf3.data.copy_():")
print(f"  buf3._version={buf3._version}, data_copy._version={data_copy._version}")
try:
    ret = ctypes.pythonapi.PyLong_Check(ctypes.py_object(buf3.untyped_storage()))
    print(f"  storage._version = {getattr(buf3.untyped_storage(), '_version', 'N/A')}")
except Exception as e:
    print(f"  Can't access storage version: {e}")

# Get the actual version from tensor internal
# Use Python internal to read int field
print(f"\n Named tokens")
# print tensor object repr
r = repr(buf3)
if '_version' in r:
    print("  repr includes _version:", r)
i = repr(data_copy)
if '_version' in i:
    print("  repr includes _version:", i)

# Final extended test: simulate what happens with real autograd
print("\n\n === Test 4: create graph, modify buffer, backward ===")
theta = torch.zeros(3, requires_grad=True)
x = torch.randn(2, 3)
buf = torch.zeros(2, 3, requires_grad=False)
print(f"theta._version={theta._version}, buf._version={buf._version}")

# Forward: use buf in graph
def forward(x, theta, buf):
    y = x + theta.unsqueeze(0)  # uses theta
    buf_copy = buf.clone()
    return y.sum() + buf_copy.sum()

loss = forward(x, theta, buf)
print(f"\nAfter forward: loss={loss}, loss_version={loss._version}")

loss.backward()
print(f"After backward: buf._version={buf._version}")
