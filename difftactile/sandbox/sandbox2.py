import taichi as ti

ti.init(
    debug=False,
    offline_cache=False,
    log_level=ti.ERROR,
    arch=ti.cpu,
)

x = ti.Vector.field(7, dtype=float, shape=(4, 1_000), needs_grad=False)
print(x.shape)
