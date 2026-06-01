import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# Data from Pair 1: u = {chol, thalach, oldpeak}, phi_u = 0.0103
# ---------------------------------------------------------
features = ['chol', 'thalach', 'oldpeak']
eq_split = [0.0034, 0.0034, 0.0034]
micro    = [-0.0044, 0.0069, 0.0077]

x = np.arange(len(features))
width = 0.30

fig, ax = plt.subplots(figsize=(3.2, 2.0))  # IEEE single-column compact

bars1 = ax.bar(x - width/2, eq_split, width,
               label='Equal-split', color='#cccccc',
               edgecolor='black', linewidth=0.5)

bars2 = ax.bar(x + width/2, micro, width,
               label='Aumann-SHAP', color='#1f77b4',
               edgecolor='black', linewidth=0.5)

# Highlight the sign-flip on chol
ax.annotate('Sign flip', xy=(0 + width/2, -0.0044), xytext=(0.5, -0.003),
            fontsize=7, ha='center', color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.8))

ax.axhline(0, color='black', linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(features, fontsize=8)
ax.set_ylabel('Within-pot share', fontsize=8)
ax.tick_params(axis='y', labelsize=7)
ax.legend(fontsize=7, frameon=False, loc='upper left')
ax.set_ylim(-0.006, 0.009)

# Remove top/right spines for clean IEEE look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.3)

# Save both formats
plt.savefig('heart_signflip.pdf', dpi=300, bbox_inches='tight')
plt.savefig('heart_signflip.png', dpi=300, bbox_inches='tight')
plt.show()

print("Saved heart_signflip.pdf and heart_signflip.png")