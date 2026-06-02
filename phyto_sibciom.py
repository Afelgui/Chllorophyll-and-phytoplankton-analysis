from __future__ import annotations

import calendar
from datetime import datetime
from pathlib import Path
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from netCDF4 import Dataset

from data_paths import AW_DIR
from getgrid import getgrid

MONTHLY_FILE_RE = re.compile(r"mth(\d{4})-(\d{2})\.nc$")


def validate_month(month: int) -> None:
    if not (1 <= month <= 12):
        raise ValueError(f"Некорректный месяц: {month}. Должен быть в диапазоне 1..12.")


def ym_to_datetime(year: int, month: int) -> datetime:
    validate_month(month)
    return datetime(year, month, 1)


def parse_monthly_filename(file_name: str) -> tuple[int, int]:
    match = MONTHLY_FILE_RE.fullmatch(file_name)
    if match is None:
        raise ValueError(f"Имя файла не соответствует шаблону mthYYYY-MM.nc: {file_name}")
    return int(match.group(1)), int(match.group(2))


def make_file_path(data_dir: Path, year: int, month: int) -> Path:
    validate_month(month)
    file_path = Path(data_dir) / f"mth{year:04d}-{month:02d}.nc"
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    return file_path


def inspect_nc(file_path: str | Path) -> None:
    with Dataset(file_path) as ds:
        print(ds)
        print("\nVARIABLES:")
        for name, var in ds.variables.items():
            units = getattr(var, "units", "")
            line = f"{name}: dims={var.dimensions}, shape={var.shape}"
            if units:
                line += f", units={units}"
            print(line)


def compute_shared_color_limits(*arrays: np.ndarray) -> tuple[float, float]:
    finite_parts: list[np.ndarray] = []
    for arr in arrays:
        arr_np = np.asarray(arr, dtype=np.float64)
        finite = arr_np[np.isfinite(arr_np)]
        if finite.size:
            finite_parts.append(finite)

    if not finite_parts:
        raise ValueError("Не удалось вычислить общую цветовую шкалу: нет конечных значений.")

    merged = np.concatenate(finite_parts)
    vmin = float(np.nanmin(merged))
    vmax = float(np.nanmax(merged))
    if vmin == vmax:
        vmax = vmin + 1e-12
    return vmin, vmax


def prepare_sibciom_3dvar(
    data_dir: Path,
    year: int,
    month: int,
    var_name: str,
    grid_base_dir: Path | None = None,
    time_index: int = 0,
) -> dict[str, object]:
    """Готовит 3D-поле SibCIOM и согласует его с вертикальной сеткой."""
    
    if grid_base_dir is None:
        grid_base_dir = AW_DIR

    file_path = make_file_path(data_dir, year, month)

    with Dataset(file_path) as ds:
        if var_name not in ds.variables:
            raise KeyError(
                f"Переменная '{var_name}' не найдена в {file_path.name}. Есть: {list(ds.variables)}"
            )

        var = ds.variables[var_name]
        units = getattr(var, "units", "")
        dims = tuple(var.dimensions)

        if var.ndim == 4:
            if dims != ("time", "knd", "jnd", "ind"):
                raise ValueError(
                    f"Ожидались dims ('time', 'knd', 'jnd', 'ind'), а получено {dims}"
                )
            data = np.ma.filled(var[time_index], np.nan).astype(np.float64)
        elif var.ndim == 3:
            if dims != ("knd", "jnd", "ind"):
                raise ValueError(f"Ожидались dims ('knd', 'jnd', 'ind'), а получено {dims}")
            data = np.ma.filled(var[:], np.nan).astype(np.float64)
        else:
            raise ValueError(f"Ожидалась 3D или 4D переменная, а получено ndim={var.ndim}, dims={dims}")

        var_3d = np.transpose(data, (2, 1, 0))

    _, _, _, z, _, _, _ = getgrid(base_dir=Path(grid_base_dir))
    z = np.asarray(z, dtype=np.float64)
    if len(z) != var_3d.shape[2]:
        raise ValueError(
            f"Число вертикальных уровней в getgrid ({len(z)}) "
            f"не совпадает с полем ({var_3d.shape[2]})."
        )
    return {
        "var_3d": var_3d,
        "levels": z,
        "units": units,
    }


def prepare_snapshot(
    data_dir: Path,
    year: int,
    month: int,
    var_name: str,
    depth_index: int = 0,
    time_index: int = 0,
) -> dict[str, object]:
    """Готовит 2D-срез SibCIOM для выбранного месяца."""
    
    
    file_path = make_file_path(data_dir, year, month)

    with Dataset(file_path) as ds:
        if var_name not in ds.variables:
            raise KeyError(
                f"Переменная '{var_name}' не найдена в {file_path.name}. Есть: {list(ds.variables)}"
            )

        var = ds.variables[var_name]
        dims = tuple(var.dimensions)
        if var.ndim == 4:
            if dims != ("time", "knd", "jnd", "ind"):
                raise ValueError(
                    f"Ожидались dims ('time', 'knd', 'jnd', 'ind'), а получено {dims}"
                )
            data = np.ma.filled(var[time_index, depth_index], np.nan).astype(np.float64)
        elif var.ndim == 3:
            if dims != ("knd", "jnd", "ind"):
                raise ValueError(f"Ожидались dims ('knd', 'jnd', 'ind'), а получено {dims}")
            data = np.ma.filled(var[depth_index], np.nan).astype(np.float64)
        else:
            raise ValueError(f"Ожидалась 3D или 4D переменная, а получено ndim={var.ndim}, dims={dims}")
        units = getattr(var, "units", "")

    return {
        "file_path": file_path,
        "data": data,
        "units": units,
    }


def plot_2d_field(
    data: np.ndarray,
    title_main: str | None = None,
    title_meta: str | None = None,
    units: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",  # Recommended: viridis, cividis, turbo, plasma
    x_name: str = "ind",
    y_name: str = "jnd",
):
    """Рисует 2D-поле с цветовой шкалой."""
    plot_data = np.asarray(data, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(plot_data, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    if title_main or title_meta:
        title_areas: list[TextArea] = []
        if title_main:
            title_areas.append(
                TextArea(
                    title_main,
                    textprops={"fontsize": 22, "fontweight": "bold", "color": "black"},
                )
            )
        if title_meta:
            title_areas.append(
                TextArea(
                    title_meta,
                    textprops={"fontsize": 12, "fontweight": "medium", "color": "#444444"},
                )
            )

        if len(title_areas) == 1:
            title_box = title_areas[0]
        else:
            title_box = HPacker(children=title_areas, align="baseline", pad=0, sep=10)

        fig.add_artist(
            AnchoredOffsetbox(
                loc="upper left",
                child=title_box,
                frameon=False,
                bbox_to_anchor=(0.08, 0.985),
                bbox_transform=fig.transFigure,
                pad=0,
                borderpad=0,
            )
        )
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(units or "value")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, ax


def plot_snapshot(
    data_dir: Path,
    year: int,
    month: int,
    var_name: str = "sal",
    depth_index: int = 0,
    time_index: int = 0,
    out_path: Path | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",  #  viridis, cividis, turbo, plasma
    title_prefix: str | None = None,
    show: bool = True,
) -> dict[str, object]:
    """Строит и, при необходимости, сохраняет 2D-срез SibCIOM."""
    
    prepared = prepare_snapshot(
        data_dir=data_dir,
        year=year,
        month=month,
        var_name=var_name,
        depth_index=depth_index,
        time_index=time_index,
    )

    month_label = calendar.month_abbr[month]
    title_main = month_label
    title_meta = f"{year:04d}  LEVEL {depth_index}"

    fig, _ = plot_2d_field(
        data=np.asarray(prepared["data"]),
        title_main=title_main,
        title_meta=title_meta,
        units=str(prepared["units"]),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return prepared


def collect_files_in_range(
    data_dir: Path,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> list[Path]:
    start_dt = ym_to_datetime(start_year, start_month)
    end_dt = ym_to_datetime(end_year, end_month)
    if start_dt > end_dt:
        raise ValueError(
            f"Начало периода позже конца: {start_year:04d}-{start_month:02d} > {end_year:04d}-{end_month:02d}"
        )

    files: list[Path] = []
    for file_path in sorted(Path(data_dir).glob("mth*.nc")):
        year, month = parse_monthly_filename(file_path.name)
        current_dt = ym_to_datetime(year, month)
        if start_dt <= current_dt <= end_dt:
            files.append(file_path)

    if not files:
        raise FileNotFoundError(
            f"Не найдено файлов mthYYYY-MM.nc в диапазоне "
            f"{start_year:04d}-{start_month:02d} .. {end_year:04d}-{end_month:02d} "
            f"в папке {data_dir}"
        )

    return files


def compute_global_limits(
    files: list[Path],
    var_name: str = "sal",
    depth_index: int = 0,
    time_index: int = 0,
    q_low: float = 1.0,
    q_high: float = 99.0,
) -> tuple[float, float]:
    if not files:
        raise ValueError("Список files пуст.")
    if q_low >= q_high:
        raise ValueError(f"Ожидалось q_low < q_high, а получено {q_low} >= {q_high}")

    lows: list[float] = []
    highs: list[float] = []

    for file_path in files:
        year, month = parse_monthly_filename(file_path.name)
        prepared = prepare_snapshot(
            data_dir=file_path.parent,
            year=year,
            month=month,
            var_name=var_name,
            depth_index=depth_index,
            time_index=time_index,
        )
        values = np.asarray(prepared["data"], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"В файле {file_path.name} нет конечных значений для {var_name}")

        lows.append(float(np.percentile(values, q_low)))
        highs.append(float(np.percentile(values, q_high)))

    vmin = float(min(lows))
    vmax = float(max(highs))
    if vmax <= vmin:
        raise ValueError(f"Некорректный диапазон цветовой шкалы: vmin={vmin}, vmax={vmax}")
    return vmin, vmax


def make_video(
    data_dir: Path,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    var_name: str = "sal",
    depth_index: int = 0,
    time_index: int = 0,
    out_video: Path = Path("video.mp4"),
    fps: float = 2.0,
    q_low: float = 1.0,
    q_high: float = 99.0,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",  # viridis, cividis, turbo, plasma
    title_prefix: str | None = None,
) -> None:
    """Собирает видео из последовательности 2D-срезов SibCIOM."""
    if fps <= 0:
        raise ValueError(f"fps должен быть > 0, а получено {fps}")

    import imageio.v2 as imageio

    files = collect_files_in_range(
        data_dir=data_dir,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
    )

    auto_vmin, auto_vmax = compute_global_limits(
        files=files,
        var_name=var_name,
        depth_index=depth_index,
        time_index=time_index,
        q_low=q_low,
        q_high=q_high,
    )
    if vmin is None:
        vmin = auto_vmin
    if vmax is None:
        vmax = auto_vmax

    frames_dir = out_video.parent / f"{out_video.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []

    for index, file_path in enumerate(files):
        year, month = parse_monthly_filename(file_path.name)
        png_path = frames_dir / f"frame_{index:04d}.png"

        plot_snapshot(
            data_dir=file_path.parent,
            year=year,
            month=month,
            var_name=var_name,
            depth_index=depth_index,
            time_index=time_index,
            out_path=png_path,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title_prefix=title_prefix,
            show=False,
        )
        frame_paths.append(png_path)

    out_video.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_video, fps=fps, codec="libx264") as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    print(f"Видео сохранено: {out_video}")
    print(f"Кадры сохранены в: {frames_dir}")


def snap_to_water(ii: int, jj: int, mask: np.ndarray) -> tuple[int, int]:
    if mask[jj, ii]:
        return ii, jj

    wet = np.argwhere(mask)
    if wet.size == 0:
        raise ValueError("В поле нет конечных значений для выбора точки.")

    d2 = (wet[:, 1] - ii) ** 2 + (wet[:, 0] - jj) ** 2
    k = int(np.argmin(d2))
    return int(wet[k, 1]), int(wet[k, 0])


def first_wet_point(mask: np.ndarray) -> tuple[int, int]:
    wet = np.argwhere(mask)
    if wet.size == 0:
        raise ValueError("В поле нет конечных значений.")
    return int(wet[0, 1]), int(wet[0, 0])


def backend_supports_interaction(backend: str | None = None) -> bool:
    """Return True for matplotlib backends that can deliver click events."""

    backend_name = (backend or matplotlib.get_backend()).lower().strip()
    if "inline" in backend_name:
        return False

    backend_leaf = backend_name.split("://", maxsplit=1)[-1].rsplit(".", maxsplit=1)[-1]
    return backend_leaf not in {
        "agg",
        "backend_agg",
        "pdf",
        "ps",
        "svg",
        "pgf",
        "template",
        "cairo",
    }


def backend_uses_jupyter_comm(backend: str | None = None) -> bool:
    """Return True for notebook backends that require an idle Jupyter kernel."""

    backend_name = (backend or matplotlib.get_backend()).lower().strip()
    return backend_name == "widget" or "ipympl" in backend_name or "nbagg" in backend_name


def running_in_notebook_kernel() -> bool:
    """Return True when code runs inside a Jupyter/IPython kernel."""

    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def ensure_blocking_selection_backend() -> tuple[str, str | None]:
    """Switch notebook widget backends to Qt for blocking point selection."""

    original_backend = matplotlib.get_backend()
    backend_name = original_backend.lower().strip()
    if not backend_uses_jupyter_comm(backend_name) or not running_in_notebook_kernel():
        return backend_name, None

    switch_errors: list[str] = []
    for candidate in ("qtagg", "qt5agg", "qtcairo"):
        try:
            plt.switch_backend(candidate)
            print(
                f"Switched matplotlib backend from {backend_name} to {matplotlib.get_backend()} "
                "for blocking point selection."
            )
            return matplotlib.get_backend().lower().strip(), original_backend
        except Exception as exc:
            switch_errors.append(f"{candidate}: {exc}")

    print(
        "Notebook widget backend detected, but Qt backend could not be enabled for blocking "
        "point selection."
    )
    for error in switch_errors:
        print(f"  {error}")
    return backend_name, None


def restore_blocking_selection_backend(backend: str | None) -> None:
    """Restore the notebook backend after a temporary Qt switch."""

    if backend is None:
        return

    try:
        plt.switch_backend(backend)
        print(f"Restored matplotlib backend to {matplotlib.get_backend()}.")
    except Exception as exc:
        print(f"Failed to restore matplotlib backend {backend!r}: {exc}")


def wait_for_interactive_selection(
    fig: matplotlib.figure.Figure,
    interactive: bool,
    done: dict[str, bool],
    prompt: str,
) -> None:
    plt.tight_layout()
    if not interactive:
        plt.show()
        return

    print(prompt)
    backend = matplotlib.get_backend().lower()
    if backend_uses_jupyter_comm(backend):
        plt.show()
        if hasattr(fig.canvas, "start_event_loop"):
            fig.canvas.start_event_loop(timeout=0)
            return
        raise RuntimeError(
            "Interactive click selection requires a matplotlib canvas with "
            "`start_event_loop()` support. Run `%matplotlib qt` or enable a Qt backend "
            "so the selector can open in a separate window."
        )

    plt.show(block=False)
    while plt.fignum_exists(fig.number) and not done["value"]:
        plt.pause(0.1)


def plot_field_from_click(
    var: np.ndarray,
    z_levels: np.ndarray,
    year: int,
    month: int,
    map_level: int = 0,
    profile_nlev: int | None = None,
    profile_max_depth: float | None = None,
    i: int | None = None,
    j: int | None = None,
    var_name: str = "sal",
    var_units: str = "",
    x_name: str = "ind",
    y_name: str = "jnd",
    map_vmin: float | None = None,
    map_vmax: float | None = None,
    map_cmap: str = "viridis",  # Recommended: viridis, cividis, turbo, plasma
) -> tuple[int, int]:
    """Интерактивная карта с профилем по клику."""
    
    backend, restore_backend = ensure_blocking_selection_backend()
    interactive = backend_supports_interaction(backend)

    if profile_nlev is not None and profile_nlev <= 0:
        raise ValueError("profile_nlev должен быть > 0")
    if profile_nlev is not None and profile_max_depth is not None:
        raise ValueError("Укажите либо profile_nlev, либо profile_max_depth, но не оба сразу.")
    if not (0 <= map_level < var.shape[2]):
        raise ValueError(f"map_level={map_level} вне диапазона 0..{var.shape[2] - 1}")

    map_data = np.asarray(var[:, :, map_level], dtype=np.float64).T.copy()
    mask = np.isfinite(map_data)
    if not np.any(mask):
        raise ValueError("На выбранном уровне нет конечных значений.")

    auto_vmin, auto_vmax = compute_shared_color_limits(map_data)
    if map_vmin is None:
        map_vmin = auto_vmin
    if map_vmax is None:
        map_vmax = auto_vmax

    if i is None or j is None:
        ii0, jj0 = first_wet_point(mask)
    else:
        ii0 = i - 1
        jj0 = j - 1
        if not (0 <= ii0 < map_data.shape[1] and 0 <= jj0 < map_data.shape[0]):
            raise ValueError(f"Начальная точка ({i}, {j}) вне границ карты.")
        ii0, jj0 = snap_to_water(ii0, jj0, mask)

    z_levels = np.asarray(z_levels, dtype=np.float64)

    def build_profile(ii: int, jj: int) -> tuple[np.ndarray, np.ndarray, float]:
        profile_full = np.asarray(var[ii, jj, :], dtype=np.float64)
        point_mask = np.isfinite(z_levels) & np.isfinite(profile_full)
        if profile_nlev is not None:
            point_mask &= np.arange(len(z_levels)) < profile_nlev
        if profile_max_depth is not None:
            point_mask &= z_levels <= profile_max_depth
        if not np.any(point_mask):
            raise ValueError(
                f"В точке ({ii + 1}, {jj + 1}) не осталось уровней для профиля после фильтрации."
            )

        depth_plot = z_levels[point_mask]
        profile = profile_full[point_mask]
        good_depth = depth_plot[np.isfinite(depth_plot) & (depth_plot > 0)]
        linthresh = float(good_depth.min()) if good_depth.size else 1.0
        return profile, depth_plot, linthresh

    profile, depth_plot, linthresh = build_profile(ii0, jj0)
    var_label = f"{var_name} ({var_units})" if var_units else var_name

    fig = plt.figure(f"{var_name} interactive selector", figsize=(16, 5))
    fig.clf()
    ax_map = fig.add_subplot(1, 3, 1)
    ax_prof = fig.add_subplot(1, 3, 2)
    ax_prof_log = fig.add_subplot(1, 3, 3)
    confirm_on_click = "ipympl" in matplotlib.get_backend().lower() or matplotlib.get_backend().lower() == "widget"

    im = ax_map.imshow(
        map_data,
        origin="lower",
        cmap=map_cmap,
        aspect="auto",
        vmin=map_vmin,
        vmax=map_vmax,
    )
    fig.colorbar(im, ax=ax_map)
    marker, = ax_map.plot(ii0, jj0, marker="*", linestyle="none", color="red", markersize=10)
    ax_map.set_xlabel(x_name)
    ax_map.set_ylabel(y_name)
    ax_map.set_title(
        f"{var_name}, {year:04d}-{month:02d}, level={map_level}, point ({ii0 + 1}, {jj0 + 1})"
    )

    prof_line, = ax_prof.plot(profile, depth_plot)
    ax_prof.set_xlabel(var_label)
    ax_prof.set_ylabel("z, m")
    ax_prof.set_title(f"{var_name} profile at point ({ii0 + 1}, {jj0 + 1})")
    ax_prof.set_ylim(np.nanmax(depth_plot), np.nanmin(depth_plot))

    prof_log_line, = ax_prof_log.plot(profile, depth_plot)
    ax_prof_log.set_yscale("symlog", linthresh=linthresh)
    ax_prof_log.set_xlabel(var_label)
    ax_prof_log.set_ylabel("z, m (symlog)")
    ax_prof_log.set_title(f"{var_name} profile symlog at point ({ii0 + 1}, {jj0 + 1})")
    ax_prof_log.set_ylim(np.nanmax(depth_plot), np.nanmin(depth_plot))

    selected = {"i": ii0 + 1, "j": jj0 + 1}
    done = {"value": False}
    connection_ids: list[int] = []

    def update(ii: int, jj: int) -> None:
        ii, jj = snap_to_water(ii, jj, mask)
        selected["i"] = ii + 1
        selected["j"] = jj + 1

        new_profile, new_depth_plot, new_linthresh = build_profile(ii, jj)
        marker.set_data([ii], [jj])
        prof_line.set_xdata(new_profile)
        prof_line.set_ydata(new_depth_plot)
        prof_log_line.set_xdata(new_profile)
        prof_log_line.set_ydata(new_depth_plot)

        ax_prof.relim()
        ax_prof.autoscale_view()
        ax_prof.set_ylim(np.nanmax(new_depth_plot), np.nanmin(new_depth_plot))

        ax_prof_log.relim()
        ax_prof_log.autoscale_view()
        ax_prof_log.set_yscale("symlog", linthresh=new_linthresh)
        ax_prof_log.set_ylim(np.nanmax(new_depth_plot), np.nanmin(new_depth_plot))

        ax_map.set_title(
            f"{var_name}, {year:04d}-{month:02d}, level={map_level}, point ({ii + 1}, {jj + 1})"
        )
        ax_prof.set_title(f"{var_name} profile at point ({ii + 1}, {jj + 1})")
        ax_prof_log.set_title(f"{var_name} profile symlog at point ({ii + 1}, {jj + 1})")
        fig.canvas.draw()
        fig.canvas.flush_events()

    if interactive:
        def on_click(event) -> None:
            if event.inaxes is not ax_map or event.xdata is None or event.ydata is None:
                return
            ii = int(round(event.xdata))
            jj = int(round(event.ydata))
            if not (0 <= ii < map_data.shape[1] and 0 <= jj < map_data.shape[0]):
                return
            update(ii, jj)
            print(f"Selected point: i={selected['i']}, j={selected['j']}")
            if confirm_on_click:
                done["value"] = True
                if hasattr(fig.canvas, "stop_event_loop"):
                    fig.canvas.stop_event_loop()

        def on_close(_event) -> None:
            done["value"] = True
            if hasattr(fig.canvas, "stop_event_loop"):
                fig.canvas.stop_event_loop()

        connection_ids.append(fig.canvas.mpl_connect("button_press_event", on_click))
        connection_ids.append(fig.canvas.mpl_connect("close_event", on_close))

    if confirm_on_click:
        selection_prompt = "Click once on the map to select a point."
    else:
        selection_prompt = (
            f"Click on the map to move the point; close the figure to confirm "
            f"({selected['i']}, {selected['j']})."
        )

    try:
        wait_for_interactive_selection(
            fig,
            interactive,
            done,
            (
                f"Using matplotlib backend {matplotlib.get_backend()}. "
                f"{selection_prompt}"
            ),
        )
    finally:
        for connection_id in connection_ids:
            fig.canvas.mpl_disconnect(connection_id)
        restore_blocking_selection_backend(restore_backend)
    return selected["i"], selected["j"]


def plot_linked_fields_from_click(
    left_var: np.ndarray,
    left_z_levels: np.ndarray,
    left_year: int,
    left_month: int,
    right_var: np.ndarray,
    right_z_levels: np.ndarray,
    right_year: int,
    right_month: int,
    map_level: int = 0,
    profile_nlev: int | None = None,
    profile_max_depth: float | None = None,
    i: int | None = None,
    j: int | None = None,
    left_var_name: str = "left",
    left_var_units: str = "",
    right_var_name: str = "right",
    right_var_units: str = "",
    x_name: str = "ind",
    y_name: str = "jnd",
    map_vmin: float | None = None,
    map_vmax: float | None = None,
    map_cmap: str = "viridis",  # Recommended: viridis, cividis, turbo, plasma
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Связанная интерактивная отрисовка двух полей."""
    
    backend, restore_backend = ensure_blocking_selection_backend()
    interactive = backend_supports_interaction(backend)

    if profile_nlev is not None and profile_nlev <= 0:
        raise ValueError("profile_nlev должен быть > 0")
    if profile_nlev is not None and profile_max_depth is not None:
        raise ValueError("Укажите либо profile_nlev, либо profile_max_depth, но не оба сразу.")
    if not (0 <= map_level < left_var.shape[2]):
        raise ValueError(f"map_level={map_level} вне диапазона левого поля 0..{left_var.shape[2] - 1}")
    if not (0 <= map_level < right_var.shape[2]):
        raise ValueError(f"map_level={map_level} вне диапазона правого поля 0..{right_var.shape[2] - 1}")

    left_map_data = np.asarray(left_var[:, :, map_level], dtype=np.float64).T.copy()
    right_map_data = np.asarray(right_var[:, :, map_level], dtype=np.float64).T.copy()
    if left_map_data.shape != right_map_data.shape:
        raise ValueError(
            "Для связанного сравнения карты должны иметь одинаковую горизонтальную форму. "
            f"Получено {left_map_data.shape} и {right_map_data.shape}."
        )

    left_mask = np.isfinite(left_map_data)
    right_mask = np.isfinite(right_map_data)
    if not np.any(left_mask):
        raise ValueError("На выбранном уровне в левом поле нет конечных значений.")
    if not np.any(right_mask):
        raise ValueError("На выбранном уровне в правом поле нет конечных значений.")

    auto_vmin, auto_vmax = compute_shared_color_limits(left_map_data, right_map_data)
    if map_vmin is None:
        map_vmin = auto_vmin
    if map_vmax is None:
        map_vmax = auto_vmax

    left_z_levels = np.asarray(left_z_levels, dtype=np.float64)
    right_z_levels = np.asarray(right_z_levels, dtype=np.float64)

    fields: list[dict[str, object]] = [
        {
            "key": "left",
            "var": np.asarray(left_var, dtype=np.float64),
            "z_levels": left_z_levels,
            "year": left_year,
            "month": left_month,
            "var_name": left_var_name,
            "var_units": left_var_units,
            "map_data": left_map_data,
            "mask": left_mask,
        },
        {
            "key": "right",
            "var": np.asarray(right_var, dtype=np.float64),
            "z_levels": right_z_levels,
            "year": right_year,
            "month": right_month,
            "var_name": right_var_name,
            "var_units": right_var_units,
            "map_data": right_map_data,
            "mask": right_mask,
        },
    ]
    confirm_on_click = "ipympl" in matplotlib.get_backend().lower() or matplotlib.get_backend().lower() == "widget"

    def build_profile(field: dict[str, object], ii: int, jj: int) -> tuple[np.ndarray, np.ndarray, float]:
        profile_full = np.asarray(field["var"][ii, jj, :], dtype=np.float64)
        z_values = np.asarray(field["z_levels"], dtype=np.float64)
        point_mask = np.isfinite(z_values) & np.isfinite(profile_full)
        if profile_nlev is not None:
            point_mask &= np.arange(len(z_values)) < profile_nlev
        if profile_max_depth is not None:
            point_mask &= z_values <= profile_max_depth
        if not np.any(point_mask):
            raise ValueError(
                f"В поле {field['var_name']!r} в точке ({ii + 1}, {jj + 1}) "
                "не осталось уровней для профиля после фильтрации."
            )

        depth_plot = z_values[point_mask]
        profile = profile_full[point_mask]
        good_depth = depth_plot[np.isfinite(depth_plot) & (depth_plot > 0)]
        linthresh = float(good_depth.min()) if good_depth.size else 1.0
        return profile, depth_plot, linthresh

    common_mask = left_mask & right_mask
    common_points = np.argwhere(common_mask)
    if common_points.size == 0:
        raise ValueError(
            "Для linked-режима не найдено ни одной горизонтальной точки, "
            "где оба поля имеют конечное значение на выбранном уровне."
        )

    def snap_to_common_point(ii: int, jj: int) -> tuple[int, int]:
        d2 = (common_points[:, 1] - ii) ** 2 + (common_points[:, 0] - jj) ** 2
        order = np.argsort(d2)
        for idx in order:
            cand_jj = int(common_points[idx, 0])
            cand_ii = int(common_points[idx, 1])
            try:
                for field in fields:
                    build_profile(field, cand_ii, cand_jj)
            except ValueError:
                continue
            return cand_ii, cand_jj

        raise ValueError(
            "Не удалось найти общую точку для linked-режима, "
            "в которой оба профиля содержат уровни после фильтрации."
        )

    if i is None or j is None:
        ii0, jj0 = snap_to_common_point(int(common_points[0, 1]), int(common_points[0, 0]))
    else:
        ii0 = i - 1
        jj0 = j - 1
        if not (0 <= ii0 < left_map_data.shape[1] and 0 <= jj0 < left_map_data.shape[0]):
            raise ValueError(f"Начальная точка ({i}, {j}) вне границ карты.")
        ii0, jj0 = snap_to_common_point(ii0, jj0)

    fig = plt.figure("linked field selector", figsize=(18, 10))
    fig.clf()
    grid = fig.add_gridspec(2, 3)

    for row, field in enumerate(fields):
        ax_map = fig.add_subplot(grid[row, 0])
        ax_prof = fig.add_subplot(grid[row, 1])
        ax_prof_log = fig.add_subplot(grid[row, 2])

        im = ax_map.imshow(
            np.asarray(field["map_data"], dtype=np.float64),
            origin="lower",
            cmap=map_cmap,
            aspect="auto",
            vmin=map_vmin,
            vmax=map_vmax,
        )
        fig.colorbar(im, ax=ax_map)

        marker, = ax_map.plot([], [], marker="*", linestyle="none", color="red", markersize=10)
        var_label = (
            f"{field['var_name']} ({field['var_units']})"
            if field["var_units"]
            else str(field["var_name"])
        )
        prof_line, = ax_prof.plot([], [])
        prof_log_line, = ax_prof_log.plot([], [])

        ax_map.set_xlabel(x_name)
        ax_map.set_ylabel(y_name)
        ax_prof.set_xlabel(var_label)
        ax_prof.set_ylabel("z, m")
        ax_prof_log.set_xlabel(var_label)
        ax_prof_log.set_ylabel("z, m (symlog)")

        field["ax_map"] = ax_map
        field["ax_prof"] = ax_prof
        field["ax_prof_log"] = ax_prof_log
        field["marker"] = marker
        field["prof_line"] = prof_line
        field["prof_log_line"] = prof_log_line

    selected: dict[str, tuple[int, int]] = {}
    done = {"value": False}
    connection_ids: list[int] = []

    def update(requested_ii: int, requested_jj: int) -> None:
        ii, jj = snap_to_common_point(requested_ii, requested_jj)
        for field in fields:
            ax_map = field["ax_map"]
            ax_prof = field["ax_prof"]
            ax_prof_log = field["ax_prof_log"]
            marker = field["marker"]
            prof_line = field["prof_line"]
            prof_log_line = field["prof_log_line"]
            new_profile, new_depth_plot, new_linthresh = build_profile(field, ii, jj)
            selected[str(field["key"])] = (ii + 1, jj + 1)

            marker.set_data([ii], [jj])
            prof_line.set_xdata(new_profile)
            prof_line.set_ydata(new_depth_plot)
            prof_log_line.set_xdata(new_profile)
            prof_log_line.set_ydata(new_depth_plot)

            ax_prof.relim()
            ax_prof.autoscale_view()
            ax_prof.set_ylim(np.nanmax(new_depth_plot), np.nanmin(new_depth_plot))

            ax_prof_log.relim()
            ax_prof_log.autoscale_view()
            ax_prof_log.set_yscale("symlog", linthresh=new_linthresh)
            ax_prof_log.set_ylim(np.nanmax(new_depth_plot), np.nanmin(new_depth_plot))

            ax_map.set_title(
                f"{field['var_name']}, {field['year']:04d}-{field['month']:02d}, "
                f"level={map_level}, point ({ii + 1}, {jj + 1})"
            )
            ax_prof.set_title(f"{field['var_name']} profile at point ({ii + 1}, {jj + 1})")
            ax_prof_log.set_title(
                f"{field['var_name']} profile symlog at point ({ii + 1}, {jj + 1})"
            )

        fig.canvas.draw()
        fig.canvas.flush_events()

    update(ii0, jj0)

    if interactive:

        def on_click(event) -> None:
            map_axes = {field["ax_map"] for field in fields}
            if event.inaxes not in map_axes or event.xdata is None or event.ydata is None:
                return
            ii = int(round(event.xdata))
            jj = int(round(event.ydata))
            if not (0 <= ii < left_map_data.shape[1] and 0 <= jj < left_map_data.shape[0]):
                return
            update(ii, jj)
            print(f"Selected shared point: i={selected['left'][0]}, j={selected['left'][1]}")
            if confirm_on_click:
                done["value"] = True
                if hasattr(fig.canvas, "stop_event_loop"):
                    fig.canvas.stop_event_loop()

        def on_close(_event) -> None:
            done["value"] = True
            if hasattr(fig.canvas, "stop_event_loop"):
                fig.canvas.stop_event_loop()

        connection_ids.append(fig.canvas.mpl_connect("button_press_event", on_click))
        connection_ids.append(fig.canvas.mpl_connect("close_event", on_close))

    if confirm_on_click:
        selection_prompt = "Click once on either map to select a shared point."
    else:
        selection_prompt = (
            f"Click on either map to move the shared point; close the figure to confirm "
            f"({selected['left'][0]}, {selected['left'][1]})."
        )

    try:
        wait_for_interactive_selection(
            fig,
            interactive,
            done,
            (
                f"Using matplotlib backend {matplotlib.get_backend()}. "
                f"{selection_prompt}"
            ),
        )
    finally:
        for connection_id in connection_ids:
            fig.canvas.mpl_disconnect(connection_id)
        restore_blocking_selection_backend(restore_backend)
    return selected["left"], selected["right"]
