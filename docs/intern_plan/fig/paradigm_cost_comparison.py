import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
})

fig, ax = plt.subplots(figsize=(9, 5.5))

paradigms = ['Standard\n(train each size)', 'Elastic\n(Stage 1)', 'All-in-One\n(Stage 2)']
x = np.arange(len(paradigms))
width = 0.22

# GPU-hours (in thousands)
# Standard: pretrain 3 models (50k each), posttrain 3 models (30k each)
# Elastic: pretrain 1 model (50k), posttrain 1 model (30k), elastic FT (10k)
# All-in-One: pretrain with elastic regularizer (55k), posttrain 1 model (30k)

pretrain = [150, 50, 55]
posttrain = [90, 30, 30]
elastic_train = [0, 35, 0]

# For Standard pretrain bar: 3 stacked portions (8B=50k, 6B=50k, 4B=50k)
standard_pretrain_portions = [50, 50, 50]

colors_pretrain = '#2563eb'
colors_posttrain = '#dc2626'
colors_elastic = '#16a34a'

# Draw bars
bars_pre = ax.bar(x - width, pretrain, width, color=colors_pretrain, edgecolor='white', linewidth=0.5, zorder=3)
bars_post = ax.bar(x, posttrain, width, color=colors_posttrain, edgecolor='white', linewidth=0.5, zorder=3)
bars_elastic = ax.bar(x + width, elastic_train, width, color=colors_elastic, edgecolor='white', linewidth=0.5, zorder=3)

# Add black separators on the Standard pretrain bar to show 3 models
# The bar goes from 0 to 150k. Split at 50k and 100k.
bar_rect = bars_pre[0]
bar_x = bar_rect.get_x()
bar_w = bar_rect.get_width()
for split_y in [50, 100]:
    ax.plot([bar_x, bar_x + bar_w], [split_y, split_y], color='black', linewidth=2, zorder=4)

# Label the three portions of the Standard pretrain bar
for i, (y_center, label) in enumerate(zip([25, 75, 125], ['4B', '6B', '8B'])):
    ax.text(bar_rect.get_x() + bar_w / 2, y_center, label,
            ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=5)

# Similarly, label the Standard posttrain bar with 3 portions
bar_rect_post = bars_post[0]
bar_x_post = bar_rect_post.get_x()
bar_w_post = bar_rect_post.get_width()
for split_y in [30, 60]:
    ax.plot([bar_x_post, bar_x_post + bar_w_post], [split_y, split_y], color='black', linewidth=2, zorder=4)
for i, (y_center, label) in enumerate(zip([15, 45, 75], ['4B', '6B', '8B'])):
    ax.text(bar_rect_post.get_x() + bar_w_post / 2, y_center, label,
            ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=5)

# Add value labels on top of bars
for bars in [bars_pre, bars_post, bars_elastic]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 2, f'{int(h)}k',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Total cost annotations
totals = [150 + 90, 50 + 30 + 35, 55 + 30]
for i, total in enumerate(totals):
    max_bar = max(pretrain[i], posttrain[i], elastic_train[i])
    ax.text(x[i], max_bar + 18, f'Total: {total}k GPU-hrs',
            ha='center', va='bottom', fontsize=10, color='#374151',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f3f4f6', edgecolor='#d1d5db', linewidth=0.8))

ax.set_xlabel('Paradigm')
ax.set_ylabel('GPU-hours (thousands)')
ax.set_title('Training Cost to Produce 3 Model Sizes (8B, 6B, 4B)')
ax.set_xticks(x)
ax.set_xticklabels(paradigms)
ax.set_ylim(0, 195)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_elements = [
    mpatches.Patch(facecolor=colors_pretrain, label='Pretrain'),
    mpatches.Patch(facecolor=colors_posttrain, label='Post-train (SFT + RL)'),
    mpatches.Patch(facecolor=colors_elastic, label='Elastic fine-tuning'),
]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig('/Users/yequan/Documents/Macro/Intern Plan/fig/paradigm_cost_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved to fig/paradigm_cost_comparison.png")
