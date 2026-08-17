import os, json, glob, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
 
LABEL_NAMES  = {"1": "ET", "2": "TC", "3": "WT"}
SKIP_METRICS = {"n_pred", "n_ref"}
 
 
# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
 
def _parse_model_name(folder: str) -> str:
    name = folder.split("_100epochs_")[0].split("Trainer")[0]
    if name != "nnUNet":
        name = name.split("nnUNet_")[1] if "nnUNet_" in name else name
    return name
 
 
def load_fold_records(base_path: str) -> pd.DataFrame:
    """
    One row per (model, fold, metric).
    Value = mean over cases in that fold — the correct CV unit of observation.
    Metrics: Mean Dice, Macro Recall, Macro Precision.
    """
    records = []
 
    summary_files = sorted(glob.glob(
        os.path.join(base_path, "*_100epochs_*", "fold_*", "validation", "summary.json")
    ))
    if not summary_files:
        raise FileNotFoundError(f"No summary files found under {base_path}")
 
    for summary_file in summary_files:
        parts      = summary_file.replace("\\", "/").split("/")
        folder     = parts[-4]
        fold_id    = parts[-3]
        model_name = _parse_model_name(folder)
 
        with open(summary_file) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] Could not decode {summary_file}")
                continue
 
        cases = data.get("metric_per_case", [])
        if not cases:
            continue
 
        case_mean_dices   = []
        case_macro_recall = []
        case_macro_prec   = []
 
        for case in cases:
            md = case.get("metrics", {})
 
            # Mean Dice across labels
            dices = [
                float(md[lbl]["Dice"])
                for lbl in ["1", "2", "3"]
                if lbl in md
                and md[lbl].get("Dice") is not None
                and not np.isnan(float(md[lbl]["Dice"]))
            ]
            if dices:
                case_mean_dices.append(np.mean(dices))
 
            # Macro Recall = TP/(TP+FN), Macro Precision = TP/(TP+FP)
            label_recalls = []
            label_precs   = []
            for lbl in ["1", "2", "3"]:
                if lbl not in md:
                    continue
                lm = md[lbl]
                tp = lm.get("TP")
                fp = lm.get("FP")
                fn = lm.get("FN")
                if None in (tp, fp, fn):
                    continue
                if (tp + fn) > 0:
                    label_recalls.append(tp / (tp + fn))
                if (tp + fp) > 0:
                    label_precs.append(tp / (tp + fp))
            if label_recalls:
                case_macro_recall.append(np.mean(label_recalls))
            if label_precs:
                case_macro_prec.append(np.mean(label_precs))
 
        for metric_name, accumulator in [
            ("Mean Dice",       case_mean_dices),
            ("Macro Recall",    case_macro_recall),
            ("Macro Precision", case_macro_prec),
        ]:
            if accumulator:
                records.append({
                    "Model":  model_name,
                    "Fold":   fold_id,
                    "Metric": metric_name,
                    "Value":  float(np.mean(accumulator)),
                })
 
    if not records:
        raise ValueError("No data collected — check summary.json structure.")
 
    return pd.DataFrame(records)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# 2.  STATISTICAL TESTING
# ═════════════════════════════════════════════════════════════════════════════
 
def rank_biserial(x, y):
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = pd.Series(np.abs(diff)).rank()
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum())
 
 
def run_statistics(df: pd.DataFrame) -> dict:
    results = {}
    models  = sorted(df["Model"].unique())
    folds   = sorted(df["Fold"].unique())
 
    for metric in df["Metric"].unique():
        mdf = df[df["Metric"] == metric]
 
        matrix = np.array([
            [mdf[(mdf["Model"] == m) & (mdf["Fold"] == f)]["Value"].values[0]
             for m in models]
            for f in folds
        ])  # shape (n_folds, n_models)
 
        friedman_stat, friedman_p = friedmanchisquare(*matrix.T)
        print(f"\n{'─'*60}")
        print(f"  {metric}  —  Friedman test")
        print(f"  chi2 = {friedman_stat:.4f},  p = {friedman_p:.4e}")
 
        meta = {"friedman_p": friedman_p, "friedman_stat": friedman_stat,
                "matrix": matrix, "models": models}
 
        if friedman_p >= 0.05:
            print("  -> No significant difference (p >= 0.05). Pairwise skipped.")
            empty = pd.DataFrame(columns=[
                "Model A", "Model B", "W", "p_raw", "p_bonf", "effect_r", "significant"
            ])
            results[metric] = (empty, meta)
            continue
 
        pairs = list(itertools.combinations(range(len(models)), 2))
        n_comp = len(pairs)
        rows = []
        for i, j in pairs:
            x, y = matrix[:, i], matrix[:, j]
            try:
                w_stat, p_raw = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                w_stat, p_raw = 0.0, 1.0
            p_bonf = min(p_raw * n_comp, 1.0)
            effect = rank_biserial(x, y)
            rows.append({
                "Model A": models[i], "Model B": models[j],
                "W": w_stat, "p_raw": p_raw, "p_bonf": p_bonf,
                "effect_r": effect, "significant": p_bonf < 0.05,
            })
            sig = "+" if p_bonf < 0.05 else " "
            print(f"  [{sig}] {models[i]:20s} vs {models[j]:20s}"
                  f"  W={w_stat:6.1f}  p_raw={p_raw:.3f}"
                  f"  p_bonf={p_bonf:.3f}  r={effect:+.3f}")
 
        results[metric] = (pd.DataFrame(rows), meta)
 
    return results
 
 
# ═════════════════════════════════════════════════════════════════════════════
# 3.  CD DIAGRAM  (maximal cliques only)
# ═════════════════════════════════════════════════════════════════════════════
 
def _maximal_cliques(sorted_models: list, stat_df: pd.DataFrame) -> list:
    """
    Return only MAXIMAL non-significant cliques — i.e. contiguous groups of
    models (in rank order) where every pairwise comparison is non-significant,
    and the group cannot be extended further in either direction.
 
    A clique that is a strict subset of a larger valid clique is suppressed,
    which is what eliminates the redundant shorter bars you observed.
    """
    n = len(sorted_models)
 
    def not_sig(a, b):
        if stat_df.empty:
            return True
        row = stat_df[
            ((stat_df["Model A"] == a) & (stat_df["Model B"] == b)) |
            ((stat_df["Model A"] == b) & (stat_df["Model B"] == a))
        ]
        return row.empty or not bool(row["significant"].values[0])
 
    # Collect all valid contiguous non-significant spans (i, j) with j > i
    valid = []
    for i in range(n):
        for j in range(i + 1, n):
            if all(not_sig(sorted_models[a], sorted_models[b])
                   for a in range(i, j + 1)
                   for b in range(a + 1, j + 1)):
                valid.append((i, j))
 
    # Keep only spans not strictly contained in a larger valid span
    maximal = []
    for span in valid:
        i, j = span
        dominated = any(
            (si <= i and sj >= j and (si, sj) != (i, j))
            for si, sj in valid
        )
        if not dominated:
            maximal.append(span)
 
    return maximal
 
 
def _cd_diagram(matrix: np.ndarray, models: list, stat_df: pd.DataFrame,
                meta: dict, metric: str, ax: plt.Axes) -> None:
    n_folds, n_models = matrix.shape
 
    # Average rank (1 = best = highest value)
    fold_ranks = np.zeros_like(matrix, dtype=float)
    for f in range(n_folds):
        order = np.argsort(matrix[f])[::-1]
        for rank, idx in enumerate(order, 1):
            fold_ranks[f, idx] = rank
    avg_ranks = fold_ranks.mean(axis=0)
 
    # Sort models left to right by average rank (best first)
    order    = np.argsort(avg_ranks)
    sorted_m = [models[i] for i in order]
    sorted_r = avg_ranks[order]
 
    cliques = _maximal_cliques(sorted_m, stat_df)
 
    # ── Draw ─────────────────────────────────────────────────────────────────
    y_dots   = 0.65
    y_bars   = 0.30
    colours  = sns.color_palette("Set2", n_models)
 
    ax.set_xlim(0.5, n_models + 0.5)
    ax.set_ylim(0, 1)
    ax.axhline(y=y_dots, color="grey", lw=0.8, ls="--", alpha=0.4)
 
    for m, r in zip(sorted_m, sorted_r):
        col = colours[models.index(m)]
        ax.plot(r, y_dots, "o", color=col, markersize=10, zorder=3)
        ax.text(r, y_dots + 0.08, f"{r:.2f}", ha="center", fontsize=7.5,
                color=col, fontweight="bold")
        ax.text(r, y_dots - 0.18, m, ha="center", va="top",
                fontsize=7, rotation=30, color=col)
 
    # One bar per maximal clique, stacked vertically if multiple
    bar_y_positions = np.linspace(y_bars, y_bars - 0.12 * (len(cliques) - 1),
                                  max(len(cliques), 1))
    for (i, j), cy in zip(cliques, bar_y_positions):
        ax.plot([sorted_r[i], sorted_r[j]], [cy, cy],
                lw=5, color="steelblue", alpha=0.75, solid_capstyle="round")
 
    friedman_p = meta.get("friedman_p", float("nan"))
    ax.set_xlabel("Average rank  (1 = best)", fontsize=9)
    ax.set_title(
        f"CD diagram — {metric}\n"
        f"Friedman p = {friedman_p:.3e}  |  bar = not significantly different",
        fontsize=8.5, fontweight="bold",
    )
    ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# 4.  BOXPLOTS
# ═════════════════════════════════════════════════════════════════════════════
 
def plot_results(df: pd.DataFrame, stat_results: dict,
                 save_dir: str | None = None) -> None:
 
    sns.set_theme(style="whitegrid", font_scale=1.05)
    models = sorted(df["Model"].unique())
 
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
 
    for metric in sorted(df["Metric"].unique()):
        mdf             = df[df["Metric"] == metric]
        stat_df, meta   = stat_results.get(metric, (pd.DataFrame(), {}))
 
        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                 gridspec_kw={"width_ratios": [1.4, 1]})
 
        # ── Left: boxplot + strip ─────────────────────────────────────────
        ax = axes[0]
        n_models      = mdf["Model"].nunique()
        model_palette = sns.color_palette("Set2", n_models)
 
        sns.boxplot(
            data=mdf, x="Model", y="Value",
            hue="Model", palette=model_palette, legend=False,
            width=0.5, linewidth=1.2,
            flierprops=dict(marker="o", markersize=5, alpha=0.6),
            ax=ax,
        )
        sns.stripplot(
            data=mdf, x="Model", y="Value",
            color="black", size=5, alpha=0.65, jitter=False, ax=ax,
        )
 
        # Significance brackets for significant pairs
        if not stat_df.empty and stat_df["significant"].any():
            model_order = [t.get_text() for t in ax.get_xticklabels()]
            tick_pos    = {m: i for i, m in enumerate(model_order)}
            y_max  = mdf["Value"].max()
            y_step = (mdf["Value"].max() - mdf["Value"].min()) * 0.06
            for level, (_, row) in enumerate(
                stat_df[stat_df["significant"]].iterrows(), 1
            ):
                ma, mb = row["Model A"], row["Model B"]
                if ma not in tick_pos or mb not in tick_pos:
                    continue
                x0, x1 = tick_pos[ma], tick_pos[mb]
                y = y_max + y_step * level
                ax.plot([x0, x0, x1, x1],
                        [y - y_step * 0.3, y, y, y - y_step * 0.3],
                        lw=1.2, color="black")
                ax.text((x0 + x1) / 2, y + y_step * 0.05, "*",
                        ha="center", va="bottom", fontsize=12)
 
        friedman_p = meta.get("friedman_p", float("nan"))
        ax.set_title(f"{metric}  (Friedman p = {friedman_p:.3e})",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Fold mean {metric}")
        ax.set_xlabel("")
 
        # Fix locator before setting labels to suppress UserWarning
        labels = [t.get_text() for t in ax.get_xticklabels()]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, ha="right", rotation=35)
 
        # ── Right: CD diagram ─────────────────────────────────────────────
        matrix = meta.get("matrix")
        if matrix is not None:
            _cd_diagram(matrix, models, stat_df, meta, metric, axes[1])
        else:
            axes[1].set_visible(False)
 
        fig.tight_layout()
 
        if save_dir:
            fname = os.path.join(save_dir, f"{metric.replace(' ', '_')}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved: {fname}")
            plt.close(fig)
        else:
            plt.show()
 
 
# ═════════════════════════════════════════════════════════════════════════════
# 5.  SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════
 
def print_and_save_stats(df: pd.DataFrame, save_dir: str | None) -> pd.DataFrame:
    stats = (
        df.groupby(["Model", "Metric"])["Value"]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    print("\n=== Mean +/- Std over folds (median) ===")
    for metric in sorted(stats["Metric"].unique()):
        print(f"\n  {metric}")
        sub = stats[stats["Metric"] == metric].sort_values("mean", ascending=False)
        for _, row in sub.iterrows():
            print(f"    {row['Model']:30s}"
                  f"  {row['mean']:.4f} +/- {row['std']:.4f}"
                  f"  (median {row['median']:.4f})")
    if save_dir:
        path = os.path.join(save_dir, "summary_stats.csv")
        stats.to_csv(path, index=False)
        print(f"\nSaved: {path}")
    return stats
 
 
# ═════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
 
def analyze_nnunet_results(base_path, save_dir=None):
    df           = load_fold_records(base_path)
    stats        = print_and_save_stats(df, save_dir)
    stat_results = run_statistics(df)
    plot_results(df, stat_results, save_dir)
 
    if save_dir:
        rows = []
        for metric, (sdf, _) in stat_results.items():
            if not sdf.empty:
                sdf = sdf.copy()
                sdf.insert(0, "Metric", metric)
                rows.append(sdf)
        if rows:
            out  = pd.concat(rows, ignore_index=True)
            path = os.path.join(save_dir, "pairwise_tests.csv")
            out.to_csv(path, index=False)
            print(f"Saved pairwise tests: {path}")
 
    return stats, df



if __name__ == "__main__":
    import argparse
    base_path = "/data/tshimanga/nnUNet_results/Dataset001_FeTS"
    save_path = "/home/tshimanga/Repositories/tokenunet/outputs/"
    analyze_nnunet_results(base_path, save_dir=save_path)