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

fig, axes = plt.subplots(
    nrows=num_plots, 
    ncols=1, 
    figsize=(8, 1.5 * num_plots),
    sharex=True, 
    gridspec_kw={'hspace': 0.4}
)

cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7]) 

for i, (label_box, file_path) in enumerate(file_paths.items()):
    print(f"Status: Processing '{label_box}' from {file_path}...")
    ax = axes[i]
    
    # Read the CSV
    df = pd.read_csv(file_path, index_col=0)
    
    # Draw the heatmap
    sns.heatmap(
        df, 
        #ax=ax, 
        cmap="Greens",     
        vmin=0,            
        #cbar=(i == 0),     
        #cbar_ax=cbar_ax if i == 0 else None,
        cbar_ax = ax
        linewidths=0       
    )
    
    # Styling the Y-axis
    ax.set_ylabel('') 
    ax.tick_params(axis='y', left=True, rotation=0, labelsize=10)
    
    # Adding the Left Box Label
    ax.text(
        x=-0.22, 
        y=0.25,  
        s=label_box, 
        transform=ax.transAxes,
        fontsize=10,
        ha='right', 
        va='center',
        bbox=dict(facecolor='lightgrey', edgecolor='black', boxstyle='square,pad=0.3', alpha=0.5)
    )

    # Styling the X-axis
    ax.set_xlabel('')
    if i == num_plots - 1:
        ax.tick_params(axis='x', bottom=False, top=False, labelbottom=True, rotation=0, labelsize=10)
    else:
        ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False)

cbar_ax.tick_params(labelsize=12, right=True, direction='out')

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
