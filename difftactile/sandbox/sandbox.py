import taichi as ti
import numpy as np

ti.init(debug=False)

x = ti.field(dtype=float, shape=(), needs_grad=True)
y = ti.field(dtype=float, shape=(), needs_grad=True)
z = ti.field(dtype=float, shape=(), needs_grad=True)

@ti.kernel
def loss(i: ti.i32):
    y[None] += 2*x[None]
    z[None] += 2*y[None]

y[None] = 0.0
x[None] = 1.0
z.grad[None] = 1.0
for i in range(10):
    y[None] = 0.0
    loss(i)
    loss.grad(i)
    print(f"i: {i}; x.grad: {x.grad[None]}; y.grad: {y.grad[None]}; z.grad: {z.grad[None]}")
