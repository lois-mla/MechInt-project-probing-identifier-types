import matplotlib
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
    figsize=(10, 1.8 * num_plots),
    sharex=True, 
    gridspec_kw={'hspace': 0.3} 
)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 

for i, (label_box, file_path) in enumerate(file_paths.items()):
    print(f"Status: Processing '{label_box}' from {file_path}...")
    ax = axes[i]
    
    # 1. READ THE CSV
    df = pd.read_csv(file_path, index_col=0)
    
    # 2. ROTATE THE DATA (Transpose rows to columns)
    df = df.T
    
    # 3. SCALE THE DATA (Convert 0.33 to 33.0 for the colorbar)
    # This checks if your max value is 1.0 or less, and multiplies by 100 if true
    if df.max().max() <= 1.0:
        df = df * 100
    
    # Draw the heatmap
    sns.heatmap(
        df, 
        ax=ax,             
        cmap="Greens",     
        vmin=0,            
        vmax=100,          
        cbar=(i == 0),     
        cbar_ax=cbar_ax if i == 0 else None,
        linewidths=0,
        cbar_kws={'ticks': [0, 20, 40, 60, 80, 100]} 
    )
    
    # Styling the Y-axis
    ax.set_ylabel('') 
    ax.tick_params(axis='y', left=False, rotation=0, labelsize=11)
    
    # Adding the Left Box Label
    ax.text(
        x=-0.25, 
        y=0.5,   
        s=label_box, 
        transform=ax.transAxes,
        fontsize=11,
        ha='right', 
        va='center',
        bbox=dict(facecolor='#f4f4f4', edgecolor='grey', boxstyle='square,pad=0.4')
    )

    # Styling the X-axis
    ax.set_xlabel('')
    if i == num_plots - 1:
        ax.tick_params(axis='x', bottom=False, labelbottom=True, rotation=0, labelsize=11)
    else:
        ax.tick_params(axis='x', bottom=False, labelbottom=False)

# Styling the Colorbar
cbar_ax.tick_params(labelsize=12, right=True, length=5, direction='out')
cbar_ax.spines['outline'].set_visible(False) 

print("Status: Plot generated, attempting to save...")

# ==========================================
# SAVE AND SHOW
# ==========================================

output_dir = "accuracies"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "accuracies_heatmap.png")

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"SUCCESS! Heatmap saved as '{save_path}'")
