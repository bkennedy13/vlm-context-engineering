import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches

# Read all result files
levels = ['level1_baseline_rag', 'level2_semantic_merged', 'level3_events',
          'level4_triview', 'level5_agentic_1']
level_names = ['L1: Baseline', 'L2: Semantic', 'L3: Events', 'L4: Tri-view', 'L5: Agentic']
colors = ['#8B7355', '#D2691E', '#4682B4', '#DC143C', '#32CD32']  # Brown, Chocolate, SteelBlue, Crimson, LimeGreen

# Task types in desired order (matching example layout)
task_order = [
    'Action Recognition',
    'Object Recognition',
    'Attribute Perception',
    'Spatial Perception',
    'Temporal Perception',
    'Information Synopsis',
    'Object Reasoning',
    'Action Reasoning',
    'Spatial Reasoning',
    'Temporal Reasoning',
    'Counting Problem',
    'OCR Problems'
]

# Collect data
data = {level: {} for level in level_names}

for i, level_file in enumerate(levels):
    with open(f'results/{level_file}.json', 'r') as f:
        result = json.load(f)
        task_stats = result['task_type_stats']

        for task in task_order:
            if task in task_stats:
                accuracy = task_stats[task]['accuracy'] * 100
                data[level_names[i]][task] = accuracy
            else:
                data[level_names[i]][task] = 0

# Create radar chart
fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(111, projection='polar')

# Number of variables
num_vars = len(task_order)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Draw background circles
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=12, color='gray')
ax.set_xticks(angles[:-1])
ax.set_xticklabels([])  # We'll add custom labels

# Add gridlines
ax.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=1)
ax.spines['polar'].set_visible(False)

# Highlight challenging tasks with colored arcs
highlight_tasks = {
    'Object Reasoning': '#FFB366',   # Orange
    'Temporal Reasoning': '#FFB366', # Orange
    'Counting Problem': '#A8D8EA',   # Light blue
}

arc_width = 0.15
for task, color in highlight_tasks.items():
    if task in task_order:
        idx = task_order.index(task)
        angle_start = angles[idx] - np.pi / num_vars
        angle_end = angles[idx] + np.pi / num_vars

        wedge = Wedge((0, 0), 1.35, np.degrees(angle_start), np.degrees(angle_end),
                     width=arc_width, facecolor=color, alpha=0.3,
                     transform=ax.transAxes, zorder=0)
        ax.add_patch(wedge)

# Plot data for each level
for i, level_name in enumerate(level_names):
    values = [data[level_name][task] for task in task_order]
    values += values[:1]  # Complete the circle

    ax.plot(angles, values, 'o-', linewidth=2.5, label=level_name,
            color=colors[i], markersize=6)
    ax.fill(angles, values, alpha=0.08, color=colors[i])

# Add task labels outside the plot
label_distance = 125  # Increased to move labels further out
for i, (angle, task) in enumerate(zip(angles[:-1], task_order)):
    # Calculate position
    x = angle
    y = label_distance

    # Adjust text alignment based on position
    ha = 'center'
    va = 'center'

    # Fine-tune alignment based on angle
    if 0 <= angle < np.pi/4 or 7*np.pi/4 <= angle < 2*np.pi:
        ha = 'left'
    elif 3*np.pi/4 < angle < 5*np.pi/4:
        ha = 'right'

    if np.pi/4 <= angle < 3*np.pi/4:
        va = 'bottom'
    elif 5*np.pi/4 <= angle < 7*np.pi/4:
        va = 'top'

    # Add text
    ax.text(x, y, task, ha=ha, va=va, fontsize=14,
            fontweight='bold', color='black')

# Add legend
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                  fontsize=14, frameon=True, fancybox=True, shadow=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

# Title with more space
plt.title('Task-Level Performance Comparison Across All Levels',
          fontsize=18, fontweight='bold', pad=50, y=1.08)

plt.tight_layout()
plt.savefig('results/task_level_comparison_radar.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✓ Radar chart saved to: results/task_level_comparison_radar.png")

# Print summary statistics
print("\nTask Performance Summary:")
print("=" * 80)
for task in task_order:
    print(f"\n{task}:")
    for level_name in level_names:
        acc = data[level_name][task]
        print(f"  {level_name}: {acc:.1f}%")
