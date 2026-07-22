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

# Conceptual relative costs (unitless)
# pretrain = posttrain = 1 unit per model
# elastic FT = slightly more than 1 unit of posttrain
unit = 1.0

pretrain = [3 * unit, unit, 1.1 * unit]  # Standard: 3 models; Elastic: 1; All-in-One: slightly more (has elastic regularizer)
posttrain = [3 * unit, unit, unit]        # Standard: 3 models; Elastic: 1; All-in-One: 1
elastic_train = [0, 1.2 * unit, 0]       # Only Elastic has this; slightly more than posttrain

colors_pretrain = '#2563eb'
colors_posttrain = '#dc2626'
colors_elastic = '#16a34a'

# Draw bars
bars_pre = ax.bar(x - width, pretrain, width, color=colors_pretrain, edgecolor='white', linewidth=0.5, zorder=3)
bars_post = ax.bar(x, posttrain, width, color=colors_posttrain, edgecolor='white', linewidth=0.5, zorder=3)
bars_elastic = ax.bar(x + width, elastic_train, width, color=colors_elastic, edgecolor='white', linewidth=0.5, zorder=3)

# Add black separators on the Standard pretrain bar (3 portions: 8B, 6B, 4B)
bar_rect = bars_pre[0]
bar_x = bar_rect.get_x()
bar_w = bar_rect.get_width()
bar_h = bar_rect.get_height()
for split_y in [unit, 2 * unit]:
    ax.plot([bar_x, bar_x + bar_w], [split_y, split_y], color='black', linewidth=2, zorder=4)

# Label the three portions of the Standard pretrain bar
for y_center, label in zip([0.5 * unit, 1.5 * unit, 2.5 * unit], ['4B', '6B', '8B']):
    ax.text(bar_rect.get_x() + bar_w / 2, y_center, label,
            ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=5)

# Standard posttrain bar: also 3 portions
bar_rect_post = bars_post[0]
bar_x_post = bar_rect_post.get_x()
bar_w_post = bar_rect_post.get_width()
for split_y in [unit, 2 * unit]:
    ax.plot([bar_x_post, bar_x_post + bar_w_post], [split_y, split_y], color='black', linewidth=2, zorder=4)
for y_center, label in zip([0.5 * unit, 1.5 * unit, 2.5 * unit], ['4B', '6B', '8B']):
    ax.text(bar_rect_post.get_x() + bar_w_post / 2, y_center, label,
            ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=5)

# Total cost annotations
totals = [pretrain[0] + posttrain[0], pretrain[1] + posttrain[1] + elastic_train[1], pretrain[2] + posttrain[2]]
labels_total = ['6x', '3.2x', '2.1x']
for i, (total, label) in enumerate(zip(totals, labels_total)):
    max_bar = max(pretrain[i], posttrain[i], elastic_train[i])
    ax.text(x[i], max_bar + 0.25, f'Total: {label}',
            ha='center', va='bottom', fontsize=11, color='#374151', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f3f4f6', edgecolor='#d1d5db', linewidth=0.8))

ax.set_xlabel('Paradigm')
ax.set_ylabel('Relative Cost')
ax.set_title('Training Cost to Produce 3 Model Sizes (8B, 6B, 4B)')
ax.set_xticks(x)
ax.set_xticklabels(paradigms)
ax.set_ylim(0, 4.2)
ax.set_yticks([])
ax.yaxis.grid(False)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

legend_elements = [
    mpatches.Patch(facecolor=colors_pretrain, label='Pretrain'),
    mpatches.Patch(facecolor=colors_posttrain, label='Post-train (SFT + RL)'),
    mpatches.Patch(facecolor=colors_elastic, label='Elastic fine-tuning'),
]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig('/Users/yequan/Documents/Macro/Intern Plan/fig/paradigm_cost_conceptual.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved to fig/paradigm_cost_conceptual.png")
