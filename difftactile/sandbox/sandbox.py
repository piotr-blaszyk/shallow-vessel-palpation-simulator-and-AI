import taichi as ti
import numpy as np

ti.init()

x = ti.Vector.field(4, dtype=int, shape=(2, 3), needs_grad=False)

print(x[0, 0].to_numpy().shape)
