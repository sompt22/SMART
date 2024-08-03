import matplotlib.pyplot as plt
import numpy as np

# Data from the table provided
distillation_levels = np.arange(0.0, 1.1, 0.1)  # Distillation levels from 0.0 to 1.0 in increments of 0.1
HOTA_scores = [47.3, 46.7, 47.1, 45.5, 46.3, 46.5, 47.1, 46.5, 47.1, 46.7, 46.1]
DetA_scores = [47.6, 47.6, 47.5, 47.3, 47.6, 47.7, 48.2, 48.1, 48.6, 48.6, 47.8]
AssA_scores = [47.3, 46.1, 47.0, 44.0, 45.3, 45.6, 46.2, 45.2, 46.3, 45.1, 44.8]
MOTA_scores = [57.0, 56.9, 56.8, 56.3, 56.8, 57.1, 57.3, 57.2, 57.3, 57.3, 56.9]
IDs_scores = [611, 626, 606, 611, 599, 616, 619, 632, 629, 676, 585]

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting the metrics except IDs on the left y-axis
ax1.plot(distillation_levels, HOTA_scores, marker='o', color='blue', label='HOTA')
ax1.plot(distillation_levels, DetA_scores, marker='o', color='red', label='DetA')
ax1.plot(distillation_levels, AssA_scores, marker='o', color='green', label='AssA')
ax1.plot(distillation_levels, MOTA_scores, marker='o', color='purple', label='MOTA')
ax1.set_xlabel('Distillation Level')
ax1.set_ylabel('Scores')
ax1.legend(loc='upper left')
ax1.grid(True)

# Creating a second y-axis for IDs scores
ax2 = ax1.twinx()
ax2.plot(distillation_levels, IDs_scores, marker='o', color='black', label='IDs')
ax2.set_ylabel('IDs Scores')
ax2.legend(loc='upper right')

plt.show()
