"""
Test to understand version counter behavior with detach and copy_()
"""
import torch

print("=== Test 1: clone().detach() maintains own version ===")
buf = torch.zeros(3, requires_grad=False)
print(f"buf._version = {buf._version}")

v1 = buf.clone().detach()
print(f"v1 = buf.clone().detach(): version = {v1._version}")

# Modify buffer via plain copy_
buf.copy_(torch.ones(3))
print(f"After buf.copy_: buf._version = {buf._version}, v1._version = {v1._version}")
print(f"  versions equal? {buf._version == v1._version}")

print("\n=== Test 2: detach() shares version ===")
buf2 = torch.zeros(3, requires_grad=False)
print(f"buf2._version = {buf2._version}")
v2 = buf2.detach()
print(f"v2 = buf2.detach(): version = {v2._version}, same id? {id(buf2) == id(v2)}")
buf2.copy_(torch.ones(3))
print(f"After buf2.copy_: buf2._version = {buf2._version}, v2._version = {v2._version}")
print(f"  versions equal? {buf2._version == v2._version}")

print("\n=== Test 3: .data.copy_() maintains version? ===")
buf3 = torch.zeros(3, requires_grad=False)
print(f"buf3._version = {buf3._version}")
v3 = buf3.clone().detach()
buf3.data.copy_(torch.ones(3))
print(f"After buf3.data.copy_: buf3._version = {buf3._version}, v3._version = {v3._version}")
print(f"  versions equal? {buf3._version == v3._version}")

print("\n=== Test 4: .data.copy_() YES increments version ===")
buf4 = torch.zeros(3, requires_grad=False)
print(f"buf4._version = {buf4._version}")
v4 = buf4.detach()  # no clone, just detach
buf4.data.copy_(torch.ones(3))
print(f"After buf4.data.copy_: buf4._version = {buf4._version}, v4._version = {v4._version}")
print(f"  versions equal? {buf4._version == v4._version}")

print("\n=== Test 5: .data.copy_() on clone() does NOT increment ===")
buf5 = torch.zeros(3, requires_grad=False)
print(f"buf5._version = {buf5._version}")
v5 = buf5.clone().detach()  # Note: clone is on buf5, then detach
buf5.data.copy_(torch.ones(3))
print(f"After buf5.data.copy_: buf5._version = {buf5._version}, v5._version = {v5._version}")
print(f"  versions equal? {buf5._version == v5._version}")

print("\nDone")
