import taichi as ti
import numpy as np

ti.init(debug=False)

x = ti.field(dtype=float, shape=(), needs_grad=True)
y = ti.field(dtype=float, shape=(), needs_grad=True)

@ti.kernel
def loss(i: ti.i32):
    y[None] += x[None]

@ti.kernel
def update_x(i: ti.i32):
    x[None] += 1.0

y[None] = 0.0
for i in range(10):
    x[None] = 0.0
    update_x(i)
    loss(i)
    loss.grad(i)
    update_x.grad(i)
    print(f"i: {i}; y: {y[None]}; x: {x[None]}; x.grad: {x.grad[None]}")

y.grad[None] = 1.0

    
