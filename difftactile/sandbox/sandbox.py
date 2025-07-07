import taichi as ti
ti.init()

N = 16

x = ti.field(dtype=ti.f32, shape=N, needs_grad=True)
y = ti.field(dtype=ti.f32, shape=N, needs_grad=False)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def func():
    for i in x:
       y[i] += x[i] ** 2
       loss[None] += y[i] ** 2

for i in range(N):
    x[i] = i

# Set the `grad` of the output variables to `1` before calling `func.grad()`.
loss.grad[None] = 1

func()
func.grad()
print(x.grad.to_numpy())
# for i in range(N):
#     assert x.grad[i] == i * 2 + 1