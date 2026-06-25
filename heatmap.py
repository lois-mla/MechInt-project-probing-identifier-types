import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

<<<<<<< HEAD


=======
>>>>>>> 0edc877fa4c213077fe241e75c246d79dd15a0bb
# ==========================================
# 2. PLOTTING SCRIPT
# ==========================================

# Dictionary mapping the label you want in the box to the CSV file path
# Update these paths to point to your actual generated CSV files
file_paths = {
    'letters mix': 'accuracies/acc_letters_mixed/acc_letters_mixed.csv',
}



num_plots = len(file_paths)

# Set up the figure and axes. 
# gridspec_kw is used to control the vertical space between heatmaps
fig, axes = plt.subplots(
    nrows=num_plots, 
    ncols=1, 
    figsize=(8, 1.5 * num_plots), # Scales figure height based on number of inputs
    sharex=True, 
    gridspec_kw={'hspace': 0.4}
)

# Add a dedicated axis on the far right for the shared colorbar
# [left, bottom, width, height]
cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7]) 

for i, (label_box, file_path) in enumerate(file_paths.items()):
    ax = axes[i]
    
    # Read the CSV. Assuming the first column contains the 'Train'/'Test' indices
    df = pd.read_csv(file_path, index_col=0)
    
    # Draw the heatmap
    sns.heatmap(
        df, 
        ax=ax, 
        cmap="Greens",     # Matches the green gradient in the image
        vmin=0,            # Force scale to start at 0
        vmax=100,          # Force scale to end at 100
        cbar=(i == 0),     # Only attach colorbar to the first loop...
        cbar_ax=cbar_ax if i == 0 else None, # ...and place it in the shared axis
        linewidths=0       # No gridlines between cells
    )
    
    # -------------------------
    # Styling the Y-axis
    # -------------------------
    ax.set_ylabel('') # Remove default ylabel
    ax.tick_params(axis='y', left=True, rotation=0, labelsize=10)
    
    # -------------------------
    # Adding the Left Box Label
    # -------------------------
    # Adjust 'x' (-0.2) if the box overlaps with your y-axis labels
    # y=0.75 aligns it roughly with the top row (Train)
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

    # -------------------------
    # Styling the X-axis
    # -------------------------
    ax.set_xlabel('')
    if i == num_plots - 1:
        # Only show bottom x-axis ticks/labels for the last plot
        ax.tick_params(axis='x', bottom=False, top=False, labelbottom=True, rotation=0, labelsize=10)
    else:
        # Hide x-axis completely for upper plots
        ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False)

# -------------------------
# Formatting the Colorbar
# -------------------------
# Set font size for the colorbar and add tick marks pointing outward
cbar_ax.tick_params(labelsize=12, right=True, direction='out')

# Optional: Add a master figure title
# plt.suptitle("Linear Probe Accuracies", y=1.05, fontsize=14)

plt.show()

# Cleanup mock files (optional)
for cat in mock_categories:
    if os.path.exists(f'mock_{cat}.csv'):

        os.remove(f'mock_{cat}.csv')

        os.remove(f'mock_{cat}.csv')

