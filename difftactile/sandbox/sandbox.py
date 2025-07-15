import taichi as ti
ti.init()

x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def func():
    y[None] += x[None] ** 2

x[None] = 3.0

y.grad[None] = 1

func()
func.grad()

