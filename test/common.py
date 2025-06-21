import random


def randshape(ndim: int, dmin: int, dmax: int, nvirtual: int) -> tuple[int | None, ...]:
    assert ndim > 0
    assert dmin > 0
    assert dmax > dmin
    assert nvirtual < ndim

    shape: list[int | None] = [
        random.randint(dmin, dmax) for _ in range(ndim - nvirtual)
    ] + [None for _ in range(nvirtual)]

    for d in reversed(range(ndim)):
        s = random.randint(0, d)
        shape[d], shape[s] = shape[s], shape[d]

    return tuple(shape)
