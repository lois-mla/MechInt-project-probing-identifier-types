import matplotlib
# This MUST come before importing pyplot or seaborn
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Status: Imports successful.")

# ==========================================
# PLOTTING SCRIPT
# ==========================================

file_paths = {
        "letters mixed": "accuracies/acc_letters_mixed/acc_letters_mixed.csv",
        "letters not mixed": "accuracies/accuracies_letters_not_mixed/accuracies_letters_not_mixed.csv" 
        }

num_plots = len(file_paths)

# Made figure slightly wider for better proportions
fig, axes = plt.subplots(
    nrows=num_plots, 
    ncols=1, 
    figsize=(10, 1.8 * num_plots),
    sharex=True, 
    gridspec_kw={'hspace': 0.3} # Tighter vertical spacing
)

# Shared colorbar axis: [left, bottom, width, height]
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 

for i, (label_box, file_path) in enumerate(file_paths.items()):
    print(f"Status: Processing '{label_box}' from {file_path}...")
    ax = axes[i]
    
    # Read the CSV
    df = pd.read_csv(file_path, index_col=0)
    
    # Draw the heatmap
    sns.heatmap(
        df, 
        ax=ax,             # FIXED: Re-enabled explicitly targeting the subplot
        cmap="Greens",     
        vmin=0,            
        vmax=100,          # Enforces the 0-100 scale
        cbar=(i == 0),     # Only attach to the first iteration
        cbar_ax=cbar_ax if i == 0 else None,
        linewidths=0,
        cbar_kws={'ticks': [0, 20, 40, 60, 80, 100]} # Match reference image ticks
    )
    
    # Styling the Y-axis
    ax.set_ylabel('') 
    # Remove the little tick marks but keep the text, horizontal rotation
    ax.tick_params(axis='y', left=False, rotation=0, labelsize=11)
    
    # Adding the Left Box Label
    ax.text(
        x=-0.25, # Push left
        y=0.5,   # Vertically centered
        s=label_box, 
        transform=ax.transAxes,
        fontsize=11,
        ha='right', 
        va='center',
        # Styled to match the neat grey boxes in the reference
        bbox=dict(facecolor='#f4f4f4', edgecolor='grey', boxstyle='square,pad=0.4')
    )

    # Styling the X-axis
    ax.set_xlabel('')
    if i == num_plots - 1:
        # Bottom plot: hide tick marks, show labels flat
        ax.tick_params(axis='x', bottom=False, labelbottom=True, rotation=0, labelsize=11)
    else:
        # Upper plots: completely hide
        ax.tick_params(axis='x', bottom=False, labelbottom=False)

# Styling the Colorbar
cbar_ax.tick_params(labelsize=12, right=True, length=5, direction='out')
cbar_ax.spines['outline'].set_visible(False) # Removes the box around the colorbar

print("Status: Plot generated, attempting to save...")

# ==========================================
# SAVE AND SHOW
# ==========================================

# SAFEGUARD: Create the 'accuracies' directory if it doesn't exist
output_dir = "accuracies"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "accuracies_heatmap.png")

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"SUCCESS! Heatmap saved as '{save_path}'")
