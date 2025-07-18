import taichi as ti
import numpy as np

ti.init()

x = ti.field(dtype=int, shape=(4,), needs_grad=False)
x_np = np.array([1, 2])

print(x)

x.from_numpy(x_np)

print(x)
