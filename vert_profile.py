from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from data_paths import AW_DIR
from getgrid import getgrid
from phyto_sibciom import validate_month


def read_chl(year: int, month: int, mask: np.ndarray, base_dir: Path) -> np.ndarray:
    """Читает chlorophyll-файл, проверяет размер и накладывает маску."""
   
    
    validate_month(month)
    file_path = base_dir / "Chl_DATA" / f"{year}" / f"chl{year:04d}-{month:02d}"
    if not file_path.exists():
        raise FileNotFoundError(f"Файл chlorophyll не найден: {file_path}")

    chl = np.fromfile(file_path, dtype=np.float64)
    expected_size = 310 * 418
    if chl.size != expected_size:
        raise ValueError(
            f"Ожидалось {expected_size} float64 в {file_path}, получено {chl.size}"
        )

    # MATLAB fread(...,[310,418],'real*8') -> Fortran order
    chl = chl.reshape((310, 418), order="F")

    if chl.shape != mask.shape:
        raise ValueError(
            f"Размер chl {chl.shape} не совпадает с mask {mask.shape}. "
            "Проверьте формат данных."
        )

    chl = chl.astype(np.float64, copy=False)
    chl[mask == 0] = np.nan
    return chl


def _read_nc_var(file_path: Path, var_name: str) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(f"NetCDF-файл не найден: {file_path}")

    with Dataset(file_path, mode="r") as ds:
        if var_name not in ds.variables:
            raise KeyError(f"Переменная {var_name!r} не найдена в {file_path}")
        arr = ds.variables[var_name][:]

    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)

    return np.asarray(arr, dtype=np.float64)


def read_daily(
    year: int,
    month: int,
    day: int,
    expected_2d_shape: tuple[int, int],
    expected_3d_shape: tuple[int, int, int],
    base_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Читает `temp` и `light_int` и приводит их к ожидаемым формам."""

    validate_month(month)
    file_sw = base_dir / "wind_data" / f"day{year}-{month:02d}-{day:02d}.nc"
    file_t = base_dir / "DAY_temp_1" / "ocn" / f"day{year}-{month:02d}-{day:02d}.nc"

    sw = np.squeeze(_read_nc_var(file_sw, "light_int"))
    sw = np.where(sw > 1e10, 0.0, sw)

    if sw.shape == expected_2d_shape[::-1]:
        sw = sw.T

    T = np.squeeze(_read_nc_var(file_t, "temp"))
    T = np.where(T > 1e10, np.nan, T)

    if T.shape == (expected_3d_shape[1], expected_3d_shape[0], expected_3d_shape[2]):
        T = np.transpose(T, (1, 0, 2))
    elif T.shape == (expected_3d_shape[2], expected_3d_shape[1], expected_3d_shape[0]):
        T = np.transpose(T, (2, 1, 0))

    if sw.shape != expected_2d_shape:
        raise ValueError(
            f"light_int shape={sw.shape}, ожидалось {expected_2d_shape}"
        )

    if T.shape != expected_3d_shape:
        raise ValueError(
            f"temp shape={T.shape}, ожидалось {expected_3d_shape}"
        )

    return T, sw


def chl_const(chl: float, depth: float, month: int) -> tuple[float, float, float, float, float, float, float]:
    """Возвращает эмпирические коэффициенты профиля хлорофилла."""
    if chl > 0:
        if depth > 50:
            if chl < 0.1:
                if 2 <= month <= 4:
                    Cb = 0.8356
                    s = 0.0026
                    Cmax = 0.945
                    zmax = 3.83
                    delz = 22.21
                    chla_zbase = 0.0285
                elif 5 <= month <= 9:
                    Cb = 0.4908
                    s = 0.0019
                    Cmax = 1.2039
                    zmax = 48.07
                    delz = 26.43
                    chla_zbase = 0.0935
                else:
                    Cb = 1.1696
                    s = 0.0045
                    Cmax = 0.113
                    zmax = 83.42
                    delz = 24.99
                    chla_zbase = 0.0427
                zbase = 110
            elif chl < 0.3:
                if 2 <= month <= 4:
                    Cb = 0.7272
                    s = 0.0009
                    Cmax = 0.8371
                    zmax = 0
                    delz = 36.2
                    chla_zbase = 0.0959
                elif 5 <= month <= 9:
                    Cb = 0.6087
                    s = 0.0026
                    Cmax = 0.9656
                    zmax = 36.05
                    delz = 27.27
                    chla_zbase = 0.1931
                else:
                    Cb = 0.6519
                    s = 0.003
                    Cmax = 0.7873
                    zmax = 2.37
                    delz = 63.03
                    chla_zbase = 0.1043
                zbase = 80
            elif chl < 0.5:
                if 2 <= month <= 4:
                    Cb = 0.4542
                    s = 0.0007
                    Cmax = 0.8127
                    zmax = 1.91
                    delz = 80.52
                    chla_zbase = 0.2764
                elif 5 <= month <= 9:
                    Cb = 0.5461
                    s = 0.0016
                    Cmax = 1.0198
                    zmax = 23.81
                    delz = 28.47
                    chla_zbase = 0.3324
                else:
                    Cb = 0.0939
                    s = 0.0001
                    Cmax = 1.4592
                    zmax = 1.34
                    delz = 66.32
                    chla_zbase = 0.2254
                zbase = 60
            elif chl < 0.7:
                if 2 <= month <= 4:
                    Cb = 0.4751
                    s = 0.0013
                    Cmax = 0.9337
                    zmax = 0
                    delz = 68.35
                    chla_zbase = 0.3904
                elif 5 <= month <= 9:
                    Cb = 0.5093
                    s = 0.0017
                    Cmax = 1.1552
                    zmax = 17.77
                    delz = 30.12
                    chla_zbase = 0.4151
                else:
                    Cb = 0.3126
                    s = 0.0013
                    Cmax = 1.3075
                    zmax = 0
                    delz = 54.03
                    chla_zbase = 0.3395
                zbase = 55
            elif chl < 1:
                Cb = 0.5449
                s = 0.0023
                Cmax = 1.1564
                zmax = 15.68
                delz = 31.69
                zbase = 50
                chla_zbase = 0.5172
            elif chl < 3:
                Cb = 0.4611
                s = 0.002
                Cmax = 1.4783
                zmax = 4.81
                delz = 35.92
                zbase = 35
                chla_zbase = 0.7841
            elif chl < 8:
                Cb = 0.487
                s = 0.0024
                Cmax = 1.7256
                zmax = 0
                delz = 31.76
                zbase = 25
                chla_zbase = 1.8078
            else:
                Cb = 0.3987
                s = 0.0019
                Cmax = 2.1463
                zmax = 6.64
                delz = 18.45
                zbase = 10
                chla_zbase = 4.3778
        else:
            if chl < 0.1:
                if 2 <= month <= 4:
                    Cb = 0.9949
                    s = 0.0113
                    Cmax = 0.255
                    zmax = 0.9621
                    delz = 0.1014
                    chla_zbase = 0.0503
                elif 5 <= month <= 9:
                    Cb = 0.0001
                    s = 1.6112
                    Cmax = 4.4054
                    zmax = 1.1616
                    delz = 0.6773
                    chla_zbase = 0.2149
                else:
                    Cb = 0.9965
                    s = 0.5444
                    Cmax = 0.7487
                    zmax = 0.8438
                    delz = 0.2959
                    chla_zbase = 0.0502
                zbase = 50
            elif chl < 0.3:
                if 2 <= month <= 4:
                    Cb = 0.9949
                    s = 0.0113
                    Cmax = 0.255
                    zmax = 0.9621
                    delz = 0.1014
                    chla_zbase = 0.1508
                elif 5 <= month <= 9:
                    Cb = 0.0001
                    s = 2.8568
                    Cmax = 4.4586
                    zmax = 1.0266
                    delz = 0.6895
                    chla_zbase = 0.3087
                else:
                    Cb = 0.9965
                    s = 0.5444
                    Cmax = 0.7487
                    zmax = 0.8438
                    delz = 0.2959
                    chla_zbase = 0.1505
                zbase = 50
            elif chl < 0.5:
                Cb = 0.0001
                s = 2.4886
                Cmax = 3.8592
                zmax = 1.0916
                delz = 0.8220
                zbase = 50
                chla_zbase = 0.5289
            elif chl < 0.7:
                Cb = 0.715
                s = 0
                Cmax = 0.8592
                zmax = 1.3961
                delz = 0.8428
                zbase = 50
                chla_zbase = 0.714
            elif chl < 1:
                Cb = 0.7990
                s = 0
                Cmax = 0.3761
                zmax = 0.7589
                delz = 0.4448
                zbase = 50
                chla_zbase = 0.9152
            elif chl < 3:
                Cb = 0.0001
                s = 1.4083
                Cmax = 2.1591
                zmax = 1.1605
                delz = 1.4467
                zbase = 35
                chla_zbase = 1.322
            elif chl < 8:
                Cb = 1.0555
                s = 0.3629
                Cmax = 0.2359
                zmax = 0.2402
                delz = 0.2483
                zbase = 25
                chla_zbase = 3.4842
            else:
                Cb = 1.0196
                s = 0.7762
                Cmax = 0.866
                zmax = 0.2144
                delz = 0.1637
                zbase = 10
                chla_zbase = 8.5078

        # Сохранено ровно как в MATLAB-коде
        zbase = 150
    else:
        Cb = 0
        s = 0
        Cmax = 0
        zmax = 0
        delz = 1
        zbase = 0
        chla_zbase = 0

    return Cb, s, Cmax, zmax, delz, chla_zbase, zbase


JERLOV_TWO_COMPONENT_COEFFS: dict[int, dict[str, float]] = {
    1: {"p": 0.58, "lambda1_cm": 0.02857, "lambda2_cm": 0.000435},
    2: {"p": 0.62, "lambda1_cm": 0.01667, "lambda2_cm": 0.0005},
    3: {"p": 0.67, "lambda1_cm": 0.01, "lambda2_cm": 0.000588},
    4: {"p": 0.77, "lambda1_cm": 0.00667, "lambda2_cm": 0.000714},
    5: {"p": 0.78, "lambda1_cm": 0.00714, "lambda2_cm": 0.001266},
}


def _normalize_inverse_length_units(units: str) -> str:
    normalized = units.strip().lower()
    aliases = {
        "cm^-1": "cm^-1",
        "cm-1": "cm^-1",
        "1/cm": "cm^-1",
        "m^-1": "m^-1",
        "m-1": "m^-1",
        "1/m": "m^-1",
    }
    if normalized not in aliases:
        raise ValueError("jerlov_lambda_units must be one of: 'cm^-1', '1/cm', 'm^-1', '1/m'")
    return aliases[normalized]


def _resolve_jerlov_coefficients(
    jerlov_type: int,
    jerlov_p: float | None,
    jerlov_lambda1: float | None,
    jerlov_lambda2: float | None,
    jerlov_lambda_units: str,
) -> tuple[float, float, float]:
    if jerlov_type not in JERLOV_TWO_COMPONENT_COEFFS:
        raise ValueError(f"Unsupported jerlov_type={jerlov_type}. Expected one of {sorted(JERLOV_TWO_COMPONENT_COEFFS)}")

    defaults = JERLOV_TWO_COMPONENT_COEFFS[jerlov_type]
    p = defaults["p"] if jerlov_p is None else float(jerlov_p)
    lambda1 = defaults["lambda1_cm"] if jerlov_lambda1 is None else float(jerlov_lambda1)
    lambda2 = defaults["lambda2_cm"] if jerlov_lambda2 is None else float(jerlov_lambda2)

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"jerlov_p must be between 0 and 1, got {p}")
    if lambda1 < 0.0 or lambda2 < 0.0:
        raise ValueError(
            f"jerlov_lambda1 and jerlov_lambda2 must be non-negative, got {lambda1} and {lambda2}"
        )

    if _normalize_inverse_length_units(jerlov_lambda_units) == "cm^-1":
        lambda1 *= 100.0
        lambda2 *= 100.0

    return p, lambda1, lambda2


def _compute_iz_at_depth(
    z_m: float,
    surface_irradiance: float,
    dt0: float,
    chl_at_depth: float,
    light_scheme: str,
    kw: float,
    kp: float,
    jerlov_coefficients_m: tuple[float, float, float] | None,
) -> float:
    if light_scheme == "beer_chl":
        attenuation = kw + kp * chl_at_depth
        return float(dt0 * surface_irradiance * np.exp(-attenuation * z_m))

    if light_scheme == "jerlov_two_component":
        if jerlov_coefficients_m is None:
            raise ValueError("jerlov_two_component requires resolved Jerlov coefficients")
        p, lambda1_m, lambda2_m = jerlov_coefficients_m
        return float(
            dt0
            * surface_irradiance
            * (
                p * np.exp(-lambda1_m * z_m)
                + (1.0 - p) * np.exp(-lambda2_m * z_m)
            )
        )

    raise ValueError(
        f"Unsupported light_scheme={light_scheme!r}. Expected 'beer_chl' or 'jerlov_two_component'"
    )


def compute_aw_fields(
    year: int = 2016,
    month: int = 6,
    day: int = 1,
    dt0: float = 3600 * 24,
    kw: float = 0.2,
    kp: float = 0.02,
    nlgr: float = 1.0,
    base_dir: Path | None = None,
    light_scheme: str = "beer_chl",
    jerlov_type: int = 1,
    jerlov_p: float | None = None,
    jerlov_lambda1: float | None = None,
    jerlov_lambda2: float | None = None,
    jerlov_lambda_units: str = "cm^-1",
):
    """Считает `chl`, `Iz` и `Aw` ."""
    if base_dir is None:
        base_dir = AW_DIR
    validate_month(month)

    if light_scheme not in {"beer_chl", "jerlov_two_component"}:
        raise ValueError("light_scheme must be 'beer_chl' or 'jerlov_two_component'")

    jerlov_coefficients_m: tuple[float, float, float] | None = None
    if light_scheme == "jerlov_two_component":
        jerlov_coefficients_m = _resolve_jerlov_coefficients(
            jerlov_type=jerlov_type,
            jerlov_p=jerlov_p,
            jerlov_lambda1=jerlov_lambda1,
            jerlov_lambda2=jerlov_lambda2,
            jerlov_lambda_units=jerlov_lambda_units,
        )

    mask, _, _, z, _, _, h = getgrid(base_dir=base_dir)
    mask = np.asarray(mask).copy()
    z = np.asarray(z, dtype=np.float64).copy()
    h = np.asarray(h, dtype=np.float64)

    im, jm = h.shape
    kb = len(z)

    mask[mask != 0] = 1

    # z[0] = 0.0
    # z[1] = 2.0
    # z[2] = 6.0

    chl0 = read_chl(year, month, mask, base_dir)
    T, I0 = read_daily(year, month, day, (im, jm), (im, jm, kb), base_dir)

    chl = np.zeros((im, jm, kb), dtype=np.float64)
    Iz = np.full((im, jm, kb), np.nan, dtype=np.float64)
    Aw = np.full((im, jm, kb), np.nan, dtype=np.float64)

    for i in range(im):
        for j in range(jm):
            Cb, s, Cmax, zmax, delz, chla_zbase, zbase = chl_const(chl0[i, j], h[i, j], month)

            k = 0
            while k < kb and z[k] <= h[i, j]:
                if chl0[i, j] > 0 and z[k] <= zbase:
                    chl_val = (Cb - s * z[k] + Cmax * np.exp(-((z[k] - zmax) / delz) ** 2)) * chla_zbase
                    chl[i, j, k] = max(chl_val, 0.0)

                Iz[i, j, k] = _compute_iz_at_depth(
                    z_m=float(z[k]),
                    surface_irradiance=float(I0[i, j]),
                    dt0=dt0,
                    chl_at_depth=float(chl[i, j, k]),
                    light_scheme=light_scheme,
                    kw=kw,
                    kp=kp,
                    jerlov_coefficients_m=jerlov_coefficients_m,
                )

                C1 = (
                    0.003
                    + 1.0154 * np.exp(0.05 * T[i, j, k])
                    * np.exp(-0.059 * Iz[i, j, k] * 1e-6)
                    * nlgr
                )
                Aw[i, j, k] = chl[i, j, k] / C1 / (2726e-9)
                k += 1

    return {
        "chl": {
            "field_key": "chl",
            "raw_3d": np.asarray(chl, dtype=np.float64),
            "levels": np.asarray(z, dtype=np.float64),
            "mask": mask,
            "var_name": "Chlorophyll-a",
            "var_units": "mg/m3",
        },
        "Iz": {
            "field_key": "Iz",
            "raw_3d": np.asarray(Iz, dtype=np.float64),
            "levels": np.asarray(z, dtype=np.float64),
            "mask": mask,
            "var_name": "Iz",
            "var_units": "uE/m2/s",
        },
        "Aw": {
            "field_key": "Aw",
            "raw_3d": np.asarray(Aw, dtype=np.float64),
            "levels": np.asarray(z, dtype=np.float64),
            "mask": mask,
            "var_name": "Aw phytoplankton",
            "var_units": "cells/m3",
        },
    }


def convert_aw_units(
    field: dict[str, np.ndarray | str],
    mode: str = "mmol",
    mmol_basis: str = "C",
    mg_c_per_cell: float = 2726e-9,
    mg_c_per_mmol_c: float = 12.011,
    n_per_c_molar: float = 16.0 / 106.0,
) -> dict[str, np.ndarray | str]:
    """Переводит `Aw` в `cells`, `mmolC/m3` или `mmolN/m3`.
    формула перевода: Aw_mmolC = Aw_cells * mg_c_per_cell / mg_c_per_mmol_c
    mmolN/m3 = mmolC/m3 * n_per_c_molar
        """
    
    field_key = str(field["field_key"])
    if field_key != "Aw":
        raise ValueError("convert_aw_units принимает только поле 'Aw'.")

    aw_cells = np.asarray(field["raw_3d"], dtype=np.float64)
    converted = {
        "field_key": field_key,
        "raw_3d": aw_cells,
        "levels": np.asarray(field["levels"], dtype=np.float64),
        "mask": np.asarray(field["mask"]),
        "var_name": "Aw phytoplankton",
        "var_units": "cells/m3",
    }

    if mode == "cells":
        return converted

    if mode != "mmol":
        raise ValueError("mode должен быть 'mmol' или 'cells'")

    aw_mmol_c = aw_cells * mg_c_per_cell / mg_c_per_mmol_c
    if mmol_basis == "C":
        converted["raw_3d"] = aw_mmol_c
        converted["var_units"] = "mmolC/m3"
        return converted
    if mmol_basis == "N":
        converted["raw_3d"] = aw_mmol_c * n_per_c_molar
        converted["var_units"] = "mmolN/m3"
        return converted

    raise ValueError("mmol_basis должен быть 'C' или 'N'")


def prepare_aw_3dvar(
    field: dict[str, np.ndarray | str],
) -> dict[str, np.ndarray | str]:
    raw_3d = np.asarray(field["raw_3d"], dtype=np.float64)
    levels = np.asarray(field["levels"], dtype=np.float64)
    mask = np.asarray(field["mask"])
    var_name = str(field["var_name"])
    var_units = str(field["var_units"])
    var_3d = np.where(mask[:, :, None] != 0, raw_3d, np.nan)

    return {
        "var_3d": var_3d,
        "levels": levels,
        "mask": mask,
        "var_name": var_name,
        "var_units": var_units,
    }


def plot_2d(
    Aw: np.ndarray,
    mask: np.ndarray,
    level: int = 0,
    i: int = 191,
    j: int = 218,
) -> None:
    """Рисует горизонтальный срез `Aw` и отмечает точку."""
    arr = Aw[:, :, level].copy()
    arr[mask == 0] = np.nan
    ii = i - 1
    jj = j - 1

    plt.figure()
    plt.contourf(arr.T, levels=100)
    if 0 <= ii < arr.shape[0] and 0 <= jj < arr.shape[1]:
        # contourf(arr.T): x -> i, y -> j
        plt.plot(ii, jj, marker="*", color="red", markersize=8)
    plt.colorbar()
    plt.title(f"Aw at level {level} (point: {i}, {j})")
    plt.tight_layout()
    plt.show()


def plot_1d(Aw: np.ndarray, z: np.ndarray, i: int = 191, j: int = 218, nlev: int = 19) -> None:
    """Рисует вертикальный профиль `Aw` в точке."""
    ii = i - 1
    jj = j - 1
    profile = Aw[ii, jj, :nlev]

    plt.figure()
    depth = np.asarray(z[:nlev], dtype=np.float64)
    plt.plot(profile, depth)
    plt.ylim(depth.max(), depth.min())
    plt.xlabel("Aw")
    plt.ylabel("z")
    plt.title(f"Aw profile at point ({i}, {j})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    result = compute_aw_fields(year=2016, month=6, day=1)
    plot_2d(result["Aw"]["raw_3d"], result["Aw"]["mask"], level=0)
    plot_1d(result["Aw"]["raw_3d"], result["Aw"]["levels"], i=191, j=218, nlev=19)
