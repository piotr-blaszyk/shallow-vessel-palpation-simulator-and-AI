import taichi as ti
ti.init()

N = 16

x = ti.field(dtype=ti.f32, shape=N, needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
b = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def func_broke_rule_1():
    # BAD: broke global data access rule #1, reading global field and before mutation is done.
    loss[None] = x[1] * b[None]
    b[None] += 100


@ti.kernel
def func_equivalent():
    loss[None] = x[1] * 10

for i in range(N):
    x[i] = i
b[None] = 10
loss.grad[None] = 1

with ti.ad.Tape(loss):
    func_broke_rule_1()
# Call func_equivalent to see the correct result
# with ti.ad.Tape(loss):
    # func_equivalent()

print(f'x.grad[1]: {x.grad[1]}')
assert x.grad[1] == 10.0