from __future__ import annotations

from pathlib import Path

import numpy as np

from data_paths import AW_DIR

MH = 310
NH = 420
KH = 40
SOURCE_DATA_DIR = AW_DIR


def _read_scalar(f, dtype):
    arr = np.fromfile(f, dtype=dtype, count=1)
    if arr.size != 1:
        raise ValueError(f"Не удалось прочитать scalar типа {dtype}")
    return arr[0]


def _read_array_f_order(f, shape, dtype):
    count = int(np.prod(shape))
    arr = np.fromfile(f, dtype=dtype, count=count)
    if arr.size != count:
        raise ValueError(
            f"Не удалось прочитать массив shape={shape}, dtype={dtype}. "
            f"Ожидалось {count} элементов, получено {arr.size}"
        )
    return arr.reshape(shape, order="F")


def getgrid(base_dir: str | Path | None = None):
    """Читает `grid2.dat`, `bound2.dat` и `top2.dat` и возвращает сетку."""
    if base_dir is None:
        base_dir = SOURCE_DATA_DIR
    else:
        base_dir = Path(base_dir)

    grid_path = base_dir / "grid2.dat"
    bound_path = base_dir / "bound2.dat"
    top_path = base_dir / "top2.dat"

    if not grid_path.exists():
        raise FileNotFoundError(f"Не найден файл: {grid_path}")
    if not bound_path.exists():
        raise FileNotFoundError(f"Не найден файл: {bound_path}")
    if not top_path.exists():
        raise FileNotFoundError(f"Не найден файл: {top_path}")

    # --- grid2.dat ---
    with open(grid_path, "rb") as f:
        _ = _read_scalar(f, np.int32)  # n
        x = _read_array_f_order(f, (MH, NH), np.float64)
        y = _read_array_f_order(f, (MH, NH), np.float64)
        z = _read_array_f_order(f, (KH,), np.float64)
        dx = _read_array_f_order(f, (MH, NH), np.float64)
        dy = _read_array_f_order(f, (MH, NH), np.float64)

    # --- bound2.dat ---
    with open(bound_path, "rb") as f:
        _ = _read_scalar(f, np.int32)  # n
        kp = _read_array_f_order(f, (MH, NH), np.int32)  # integer*4
        _ = _read_array_f_order(f, (MH, NH), np.int16)  # kp0
        _ = _read_array_f_order(f, (MH, NH), np.int16)  # kp0
        mask = _read_array_f_order(f, (MH, NH), np.int16)  # integer*2

    # --- top2.dat ---
    with open(top_path, "rb") as f:
        _ = _read_scalar(f, np.int32)  # n
        h = _read_array_f_order(f, (MH, NH), np.float64)

    h = h.astype(np.float64, copy=False)
    h[mask == 0] = np.nan

    # MATLAB:
    # x=x(2:end-1,2:end-1);
    # y=y(2:end-1,2:end-1);
    # dx=dx(2:end-1,2:end-1);
    # dy=dy(2:end-1,2:end-1);
    # z=z(2:end-1);
    # h=h(2:end-1,2:end-1);
    # mask=mask(2:end-1,2:end-1);
    # kp=kp(2:end-1,2:end-1);

    z = z[1:-1]
    x = x[:, 1:-1]
    y = y[:, 1:-1]
    dx = dx[:, 1:-1]
    dy = dy[:, 1:-1]
    h = h[:, 1:-1]
    mask = mask[:, 1:-1]
    kp = kp[:, 1:-1]

    return mask, x, y, z, dx, dy, h


if __name__ == "__main__":
    mask, x, y, z, dx, dy, h = getgrid()
    print("mask:", mask.shape, mask.dtype)
    print("x   :", x.shape, x.dtype)
    print("y   :", y.shape, y.dtype)
    print("z   :", z.shape, z.dtype)
    print("dx  :", dx.shape, dx.dtype)
    print("dy  :", dy.shape, dy.dtype)
    print("h   :", h.shape, h.dtype)
    print("nan in h:", np.isnan(h).sum())
