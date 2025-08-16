def num_markers_in_ring(k):
    if k == 0:
        return 1
    else:
        return k * 6

rings = list(range(0, 7))
rings = [num_markers_in_ring(x) for x in rings]
print(sum(rings))
