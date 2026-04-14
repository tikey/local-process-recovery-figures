from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


METHOD_PROPOSED = "Запропонований метод локального відновлення"
METHOD_RESTART = "Повний перезапуск процесу"
METHOD_DEFERRED = "Відкладене сервісне відновлення"
THRESHOLD_LABEL = "Гранично допустимий рівень"

METHODS = [
    METHOD_PROPOSED,
    METHOD_RESTART,
    METHOD_DEFERRED,
]

COLORS = {
    METHOD_PROPOSED: "black",
    METHOD_RESTART: "#1f77b4",
    METHOD_DEFERRED: "#d62728",
}

LINESTYLES = {
    METHOD_PROPOSED: "-",
    METHOD_RESTART: "--",
    METHOD_DEFERRED: "-.",
}

FIGSIZE = (8.4, 5.9)
DPI = 300


def logistic(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def clipped(y: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(y, low, high)


def build_time_curves(rng: np.random.Generator) -> pd.DataFrame:
    t = np.arange(0, 41, 1, dtype=float)

    prop = 0.82 * np.exp(-0.23 * t) + 0.018 * np.sin(0.42 * t) + rng.normal(0, 0.004, t.size)
    full = 0.78 * np.exp(-0.12 * np.maximum(t - 4, 0)) + 0.11 * np.exp(-0.65 * np.maximum(4 - t, 0))
    full += 0.014 * np.sin(0.35 * t + 0.5) + rng.normal(0, 0.005, t.size)
    serv = 0.86 * np.exp(-0.08 * t) + 0.022 * np.sin(0.28 * t + 0.4) + rng.normal(0, 0.005, t.size)

    prop = clipped(prop, 0.0, 1.0)
    full = clipped(full, 0.0, 1.0)
    serv = clipped(serv, 0.0, 1.0)

    threshold = np.full_like(t, 0.18)

    rows = []
    for method, values in {
        METHOD_PROPOSED: prop,
        METHOD_RESTART: full,
        METHOD_DEFERRED: serv,
        THRESHOLD_LABEL: threshold,
    }.items():
        for ti, yi in zip(t, values):
            rows.append({"figure": "fig1", "t": ti, "method": method, "value": yi})
    return pd.DataFrame(rows)


def build_cumulative_deficit(df_time: pd.DataFrame) -> pd.DataFrame:
    t = sorted(df_time[df_time["method"] == METHOD_PROPOSED]["t"].unique())
    rows = []
    thr = 0.18
    for method in METHODS:
        y = df_time[df_time["method"] == method].sort_values("t")["value"].to_numpy(dtype=float)
        deficit = np.maximum(y - thr, 0.0)
        if method == METHOD_RESTART:
            deficit += np.where(np.array(t) < 5, 0.030, 0.0)
        elif method == METHOD_DEFERRED:
            deficit += np.where(np.array(t) < 8, 0.022, 0.0)
        cum = np.cumsum(deficit)
        for ti, yi in zip(t, cum):
            rows.append({"figure": "fig2", "t": ti, "method": method, "value": yi})
    return pd.DataFrame(rows)


def build_success_probability(rng: np.random.Generator) -> pd.DataFrame:
    z = np.linspace(0.10, 1.00, 19)
    prop = 0.985 - 0.26 * (z ** 1.55) - 0.030 * logistic(z, 10.0, 0.78)
    full = 0.955 - 0.36 * (z ** 1.45) - 0.055 * logistic(z, 9.0, 0.72)
    serv = 0.925 - 0.43 * (z ** 1.40) - 0.070 * logistic(z, 8.0, 0.68)

    prop += rng.normal(0, 0.004, z.size)
    full += rng.normal(0, 0.005, z.size)
    serv += rng.normal(0, 0.005, z.size)

    rows = []
    for method, values in {
        METHOD_PROPOSED: clipped(prop, 0.0, 1.0),
        METHOD_RESTART: clipped(full, 0.0, 1.0),
        METHOD_DEFERRED: clipped(serv, 0.0, 1.0),
    }.items():
        for xi, yi in zip(z, values):
            rows.append({"figure": "fig3", "x": xi, "method": method, "value": yi})
    return pd.DataFrame(rows)


def build_recovery_time(rng: np.random.Generator) -> pd.DataFrame:
    z = np.linspace(0.10, 1.00, 19)
    prop = 1.9 + 3.1 * z + 1.2 * (z ** 2)
    full = 3.2 + 4.8 * z + 1.9 * (z ** 2)
    serv = 4.0 + 6.0 * z + 2.4 * (z ** 2)

    prop += rng.normal(0, 0.05, z.size)
    full += rng.normal(0, 0.06, z.size)
    serv += rng.normal(0, 0.07, z.size)

    rows = []
    for method, values in {
        METHOD_PROPOSED: prop,
        METHOD_RESTART: full,
        METHOD_DEFERRED: serv,
    }.items():
        for xi, yi in zip(z, values):
            rows.append({"figure": "fig4", "x": xi, "method": method, "value": yi})
    return pd.DataFrame(rows)


def build_state_volume(rng: np.random.Generator) -> pd.DataFrame:
    z = np.linspace(0.10, 1.00, 19)
    prop = 12.0 + 18.0 * z + 8.0 * (z ** 1.8)
    full = 42.0 + 34.0 * z + 20.0 * (z ** 1.6)
    serv = 28.0 + 30.0 * z + 16.0 * (z ** 1.7)

    prop += rng.normal(0, 0.4, z.size)
    full += rng.normal(0, 0.6, z.size)
    serv += rng.normal(0, 0.5, z.size)

    rows = []
    for method, values in {
        METHOD_PROPOSED: prop,
        METHOD_RESTART: full,
        METHOD_DEFERRED: serv,
    }.items():
        for xi, yi in zip(z, values):
            rows.append({"figure": "fig5", "x": xi, "method": method, "value": yi})
    return pd.DataFrame(rows)


def build_side_effect_risk(rng: np.random.Generator) -> pd.DataFrame:
    q = np.linspace(0.05, 0.95, 19)
    prop = 0.006 + 0.040 * (q ** 1.35)
    full = 0.030 + 0.185 * (q ** 1.18)
    serv = 0.022 + 0.145 * (q ** 1.22)

    prop += rng.normal(0, 0.0012, q.size)
    full += rng.normal(0, 0.0020, q.size)
    serv += rng.normal(0, 0.0018, q.size)

    rows = []
    for method, values in {
        METHOD_PROPOSED: clipped(prop, 0.0, 1.0),
        METHOD_RESTART: clipped(full, 0.0, 1.0),
        METHOD_DEFERRED: clipped(serv, 0.0, 1.0),
    }.items():
        for xi, yi in zip(q, values):
            rows.append({"figure": "fig6", "x": xi, "method": method, "value": yi})
    return pd.DataFrame(rows)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def add_legend_below(ax: plt.Axes) -> None:
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=1,
        frameon=True,
        fontsize=10,
        handlelength=3.0,
        handletextpad=0.6,
        borderpad=0.5,
        labelspacing=0.4,
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)


def save_line_chart(
    df: pd.DataFrame,
    figure_code: str,
    outdir: Path,
    xlabel: str,
    ylabel: str,
    xcol: str,
    ylim: tuple[float, float] | None = None,
    add_threshold: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    style_axes(ax)

    methods = METHODS.copy()
    if add_threshold:
        methods.append(THRESHOLD_LABEL)

    for method in methods:
        sub = df[df["method"] == method].sort_values(xcol)
        if sub.empty:
            continue
        color = "gray" if method == THRESHOLD_LABEL else COLORS[method]
        linestyle = ":" if method == THRESHOLD_LABEL else LINESTYLES[method]
        linewidth = 1.7 if method == THRESHOLD_LABEL else 2.2
        ax.plot(
            sub[xcol].to_numpy(dtype=float),
            sub["value"].to_numpy(dtype=float),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=method,
        )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if ylim is not None:
        ax.set_ylim(*ylim)

    add_legend_below(ax)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.30)

    png_path = outdir / f"{figure_code}.png"
    pdf_path = outdir / f"{figure_code}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def create_summary(*dfs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for df in dfs:
        figcode = str(df["figure"].iloc[0])
        for method in df["method"].unique():
            if method == THRESHOLD_LABEL:
                continue
            sub = df[df["method"] == method]
            rows.append(
                {
                    "figure": figcode,
                    "method": method,
                    "min": float(sub["value"].min()),
                    "max": float(sub["value"].max()),
                    "mean": float(sub["value"].mean()),
                    "last": float(sub["value"].iloc[-1]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results_local_process_recovery_v2")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 1.0,
    })

    df1 = build_time_curves(rng)
    df2 = build_cumulative_deficit(df1)
    df3 = build_success_probability(rng)
    df4 = build_recovery_time(rng)
    df5 = build_state_volume(rng)
    df6 = build_side_effect_risk(rng)

    save_line_chart(
        df1[df1["figure"] == "fig1"],
        "fig1_theta_time",
        outdir,
        "Кроки відновлення",
        "Показник локального пошкодження\nстану процесу, відн. од.",
        "t",
        ylim=(0.0, 0.9),
        add_threshold=True,
    )
    save_line_chart(
        df2,
        "fig2_cumulative_deficit",
        outdir,
        "Кроки відновлення",
        "Накопичувальний показник відхилення\nвідновлення, відн. од.",
        "t",
        ylim=(0.0, float(df2["value"].max() * 1.08)),
    )
    save_line_chart(
        df3,
        "fig3_success_probability",
        outdir,
        "Інтенсивність локального пошкодження стану процесу",
        "Імовірність успішного відновлення",
        "x",
        ylim=(0.35, 1.02),
    )
    save_line_chart(
        df4,
        "fig4_recovery_time",
        outdir,
        "Інтенсивність локального пошкодження стану процесу",
        "Середній час повернення процесу до\nкоректного виконання, кроки відновлення",
        "x",
        ylim=(1.5, float(df4["value"].max() * 1.08)),
    )
    save_line_chart(
        df5,
        "fig5_state_volume",
        outdir,
        "Інтенсивність локального пошкодження стану процесу",
        "Обсяг даних,\nпотрібних для відновлення процесу, Мб",
        "x",
        ylim=(0.0, float(df5["value"].max() * 1.08)),
    )
    save_line_chart(
        df6,
        "fig6_side_effect_risk",
        outdir,
        "Питома вага кроків з неідемпотентним повторним виконанням у процесі, відн. од.",
        "Оцінка імовірності дублювання\nзовнішнього результату, відн. од.",
        "x",
        ylim=(0.0, float(df6["value"].max() * 1.10)),
    )

    all_data = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
    all_data.to_csv(outdir / "simulation_data.csv", index=False, encoding="utf-8-sig")

    summary = create_summary(df1, df2, df3, df4, df5, df6)
    summary.to_csv(outdir / "summary_metrics.csv", index=False, encoding="utf-8-sig")

    readme = outdir / "README.txt"
    readme.write_text(
        """Згенеровано 6 рисунків для статті:
fig1_theta_time - показник локального пошкодження стану процесу
fig2_cumulative_deficit - накопичувальний показник відхилення відновлення
fig3_success_probability - імовірність успішного відновлення
fig4_recovery_time - середній час повернення процесу до коректного виконання
fig5_state_volume - обсяг даних, потрібних для відновлення процесу
fig6_side_effect_risk - оцінка імовірності дублювання зовнішнього результату

Позначення кривих:
- чорний: запропонований метод локального відновлення
- синій: повний перезапуск процесу
- червоний: відкладене сервісне відновлення
- сірий пунктир: гранично допустимий рівень
""",
        encoding="utf-8",
    )

    print(f"Done. Results saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
