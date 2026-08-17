import pandas as pd
import matplotlib.pyplot as plt
import io

# 1. Load the data

df = pd.read_csv("/home/tshimanga/Repositories/tokenunet/outputs/profiling_results.csv")

def shorten_name(name):
    name = name.replace("nnUNet_", "").replace("Trainer_100epochs", "")
    return "Baseline (NoToken)" if name == "NoTokenUNet" else name

df['short_name'] = df['trainer'].apply(shorten_name)

def get_style(name):
    if "Baseline" in name:       return "#7f8c8d", "o"   # Muted Gray Circle
    if "Swin" in name:           return "#e74c3c", "D"   # Accent Red Diamond
    if "8" in name:
        if "Attn" in name:       return "#e67e22", "^"   # Orange Triangle
        if "MLP" in name:        return "#f39c12", "s"   # Amber Square
        return "#d35400", "o"                            # Rust Circle
    if "32" in name:
        if "Long" in name:       return "#2c3e50", "X"   # Dark Slate Cross
        if "Attn" in name:       return "#2980b9", "^"   # Blue Triangle
        if "MLP" in name:        return "#3498db", "s"   # Light Blue Square
        return "#1abc9c", "o"                            # Teal Circle
    return "#34495e", "o"

df['color'] = df['short_name'].apply(lambda x: get_style(x)[0])
df['marker'] = df['short_name'].apply(lambda x: get_style(x)[1])

# Hand-crafted collision-free offsets for close-range points in Panel B
offsets_refined = {
    'Baseline (NoToken)': (-50, -25),
    '32TokenUNet': (-60, 25),
    '32MLPTokenUNet': (-75, 10),
    '32AttnLongTokenUNet': (20, 10),
    '8TokenUNet': (20, -15),
    '8MLPTokenUNet': (-30, 25),
    '32AttnTokenUNet': (25, -20),
    '8AttnTokenUNet': (25, 20)
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. REUSABLE PLOTTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_efficiency_plot(df, x_col, y_col, x_label, y_label, title_prefix, out_filename):
    """Generates a 2-panel efficiency plot for given X and Y metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))

    # --- PANEL A: Global View ---
    for idx, row in df.iterrows():
        ax1.scatter(
            row[x_col], row[y_col], 
            s=row['n_params'] * 150, color=row['color'], 
            alpha=0.7, edgecolors='black', linewidth=1, marker=row['marker']
        )
        # Annotations for outliers/clusters
        if "Swin" in row['short_name']:
            ax1.annotate(f"SwinUNETR\n(15.71M params)", (row[x_col], row[y_col]),
                        xytext=(-105, -15), textcoords='offset points', fontsize=10, weight='bold',
                        arrowprops=dict(arrowstyle="->", color='black', lw=0.8))
        elif "Baseline" in row['short_name']:
            ax1.annotate("TokenUNet\nCluster", (row[x_col], row[y_col]),
                        xytext=(40, 20), textcoords='offset points', fontsize=11, weight='bold',
                        arrowprops=dict(arrowstyle="->", color='black', lw=0.8, connectionstyle="arc3,rad=-0.2"))

    ax1.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax1.set_ylabel(y_label, fontsize=12, labelpad=10)
    ax1.set_title(f'A. Global {title_prefix} Landscape', fontsize=14, weight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Info box on Panel A
    ax1.text(0.05, 0.93, "Bubble Area $\propto$ Parameter Count\n\n• TokenUNet Variants: 1.67M - 2.74M\n• SwinUNETR Baseline: 15.71M", 
             transform=ax1.transAxes, fontsize=10.5, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dcdde1', alpha=0.9))

    # --- PANEL B: Granular View (TokenUNet only) ---
    token_df = df[df['short_name'] != 'SwinUNETR'].copy()

    for idx, row in token_df.iterrows():
        ax2.scatter(
            row[x_col], row[y_col], 
            s=row['n_params'] * 500, color=row['color'], 
            alpha=0.8, edgecolors='black', linewidth=1.2, marker=row['marker']
        )

    for idx, row in token_df.iterrows():
        name = row['short_name']
        ox, oy = offsets_refined.get(name, (20, 20))
        ax2.annotate(
            f"{name}\n{row['n_params']:.2f}M params", 
            (row[x_col], row[y_col]),
            xytext=(ox, oy), textcoords='offset points', fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color='#34495e', lw=0.8, alpha=0.7)
        )

    ax2.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax2.set_ylabel(y_label, fontsize=12, labelpad=10)
    ax2.set_title(f'B. Granular Trade-offs: {title_prefix}', fontsize=14, weight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Dynamically calculate tight boundaries for Panel B with a 15% margin
    x_range = token_df[x_col].max() - token_df[x_col].min()
    y_range = token_df[y_col].max() - token_df[y_col].min()
    
    # Avoid zero-division if all points land on the exact same axis value
    x_margin = x_range * 0.15 if x_range > 0 else 2.0
    y_margin = y_range * 0.15 if y_range > 0 else 20.0

    ax2.set_xlim(token_df[x_col].min() - x_margin, token_df[x_col].max() + x_margin)
    ax2.set_ylim(token_df[y_col].min() - y_margin, token_df[y_col].max() + y_margin)

    plt.tight_layout()
    plt.savefig(out_filename, dpi=300)
    plt.close()
    print(f"Saved: {out_filename}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERATE ALL THREE PLOTS
# ─────────────────────────────────────────────────────────────────────────────
base_path = "/home/tshimanga/Repositories/tokenunet/outputs/"

# Plot 1: Throughput (Inference Time vs Training Time)
generate_efficiency_plot(
    df, 
    x_col="inf_mean_ms", y_col="train_mean_ms", 
    x_label="Inference Latency per Sample (ms)", y_label="Training Step Duration (ms)", 
    title_prefix="Computational Throughput",
    out_filename=base_path + "throughput_time_vs_time.png"
)

# Plot 2: Inference Profiling (Inference Time vs Inference Peak Memory)
generate_efficiency_plot(
    df, 
    x_col="inf_mean_ms", y_col="inf_peak_mem_mb", 
    x_label="Inference Latency per Sample (ms)", y_label="Peak Inference Memory (MB)", 
    title_prefix="Inference Efficiency",
    out_filename=base_path + "throughput_infTime_vs_infMem.png"
)

# Plot 3: Training Profiling (Training Time vs Training Peak Memory)
generate_efficiency_plot(
    df, 
    x_col="train_mean_ms", y_col="train_peak_mem_mb", 
    x_label="Training Step Duration (ms)", y_label="Peak Training Memory (MB)", 
    title_prefix="Training Efficiency",
    out_filename=base_path + "throughput_trainTime_vs_trainMem.png"
)