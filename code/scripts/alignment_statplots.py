import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
OUTPUT_PATH = "/home/tshimanga/Repositories/tokenunet/outputs/"
READ_PATH = OUTPUT_PATH + "attention_alignment/"

df = pd.read_csv(READ_PATH + "alignment_metrics_summary.csv")
df['short_name'] = df['Architecture'].apply(lambda x: x.replace("nnUNet_", "").replace("Trainer_100epochs", ""))

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
regions = ['ET', 'TC', 'WT']
region_titles = {'ET': 'Enhancing Tumor (ET)', 'TC': 'Tumor Core (TC)', 'WT': 'Whole Tumor (WT)'}

unique_archs = df['short_name'].unique()
cmap = plt.cm.get_cmap('tab10', len(unique_archs))
arch_colors = {arch: cmap(i) for i, arch in enumerate(unique_archs)}

for i, region in enumerate(regions):
    ax = axes[i]
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    
    for arch in unique_archs:
        row = df[df['short_name'] == arch].iloc[0]
        tl_val = row[f'TL_{region}_Best']
        tf_val = row[f'TF_{region}_Best']
        
        ax.plot(['TokenLearner\n(Gather)', 'TokenFuser\n(Broadcast)'], [tl_val, tf_val], 
                color=arch_colors[arch], linestyle='-', linewidth=2, alpha=0.7, zorder=2)
        ax.scatter('TokenLearner\n(Gather)', tl_val, color=arch_colors[arch], s=80, edgecolors='black', zorder=3)
        ax.scatter('TokenFuser\n(Broadcast)', tf_val, color=arch_colors[arch], s=80, edgecolors='black', zorder=3, label=arch if i == 2 else "")

    ax.set_title(region_titles[region], fontsize=13, weight='bold', pad=12)
    ax.set_ylim(-0.02, 0.7)
    if i == 0:
        ax.set_ylabel('Best Token Spatial Alignment (Dice Score)', fontsize=12, labelpad=10)
    ax.tick_params(axis='both', labelsize=11)

axes[2].legend(title='Architecture', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=11)
plt.suptitle('Spatial Attention Encoding vs Label Decoding', fontsize=15, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_PATH+'token_alignment_gather_broadcast.png', dpi=300, bbox_inches='tight')
plt.close()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
x = np.arange(len(unique_archs))
width = 0.25

# Panel A: TokenLearner
ax1.bar(x - width, df['TL_ET_Best'], width, label='Enhancing Tumor (ET)', color='#e74c3c', edgecolor='black', alpha=0.8)
ax1.bar(x, df['TL_TC_Best'], width, label='Tumor Core (TC)', color='#2ecc71', edgecolor='black', alpha=0.8)
ax1.bar(x + width, df['TL_WT_Best'], width, label='Whole Tumor (WT)', color='#3498db', edgecolor='black', alpha=0.8)
ax1.set_title('A. TokenLearner (Gathering Phase) - Maximum Spatial Alignment', fontsize=13, weight='bold', pad=10)
ax1.set_ylabel('Best Token Dice Score', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_ylim(0, 0.7)

# Panel B: TokenFuser
ax2.bar(x - width, df['TF_ET_Best'], width, label='Enhancing Tumor (ET)', color='#e74c3c', edgecolor='black', alpha=0.8)
ax2.bar(x, df['TF_TC_Best'], width, label='Tumor Core (TC)', color='#2ecc71', edgecolor='black', alpha=0.8)
ax2.bar(x + width, df['TF_WT_Best'], width, label='Whole Tumor (WT)', color='#3498db', edgecolor='black', alpha=0.8)
ax2.set_title('B. TokenFuser (Broadcasting Phase) - Maximum Spatial Alignment', fontsize=13, weight='bold', pad=10)
ax2.set_ylabel('Best Token Dice Score', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_ylim(0, 0.7)

ax2.set_xticks(x)
ax2.set_xticklabels(unique_archs, rotation=15, ha='right', fontsize=11)
plt.suptitle('Comparison of Maximum Anatomical Specialization Across Architectures', fontsize=16, weight='bold', y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_PATH+'token_alignment_bars.png', dpi=300, bbox_inches='tight')
plt.close()