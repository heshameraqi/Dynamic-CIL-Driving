import matplotlib.pyplot as plt
import os
import sys
import re
import numpy as np

# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
"""
files = ['/media/heraqi/data/heraqi/int-end-to-end-ad/models/F034BF_AUC2_data_1cam_26/losses.txt',
         '/media/heraqi/data/heraqi/int-end-to-end-ad/models/F034BF_AUC2_data_1cam-pgm_2/losses.txt'
         ]       
method_names = ['Our data - 1cam', 'Our data - 1cam & LiDAR']
"""
# Results on smaller our dataset
files = ['/media/heraqi/data/heraqi/int-end-to-end-ad/models/F3F1E4_CIL_data_3/losses.txt',
         '/media/heraqi/data/heraqi/int-end-to-end-ad/models/F3F1E4_AUC2_data_1cam_27/losses.txt'
         '/media/heraqi/data/heraqi/int-end-to-end-ad/models/F3F1E4_AUC2_data_3cams/losses.txt',
         '/media/heraqi/data/heraqi/int-end-to-end-ad/models/F3F1E4_AUC2_data_3cams-pgm_5/losses.txt']
method_names = ['CIL data', 'AUC data - 1cam', 'AUC data - 3cams', 'AUC data - 3cams & LiDAR']
save_figure_directory = "/home/heraqi/scripts/int-end-to-end-ad-carla-valeo/_benchmarks_results/"

# ------------------------------------------------------------------------------
# Parse numbers
# ------------------------------------------------------------------------------
pattern = re.compile(r'^Steps: \d+, Epoch: \d+, Minibatch: \d+, Train Loss: (\d+\.\d+), Validation Loss: (\d+\.\d+), Learning Rate: (\d+\.\d+)')
data = []
for file in files:
    lines = open(file, "r")
    data.append([])
    for line in lines:
        groups = re.search(pattern, line)
        train_loss = groups.group(1)
        val_loss = groups.group(2)
        learning_rate = groups.group(3)

        data[-1].append([float(train_loss), float(val_loss)])

# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 4))
cmap = plt.get_cmap('jet_r')
N = len(data)
used_xs = []
for i in range(N):
    color = cmap(float(i)/N)
    train_losses = [data[i][x][0] for x in range(len(data[i]))]
    val_losses = [data[i][x][1] for x in range(len(data[i]))]
    plt.plot(np.arange(1, len(train_losses)+1), train_losses, marker='o', markersize=2, c=color, label=method_names[i] + ' (train data)')
    plt.plot(np.arange(1, len(val_losses)+1), val_losses, c=color, label=method_names[i] + ' (validation data)')
    x = np.argmin(val_losses)+1
    
    text_y_loc = 0.035
    # if previous model had same stop epoch
    if x in used_xs:
        x += 0.1
        text_y_loc = 0.028
    used_xs.append(x)
    plt.axvline(x=x, c=color)
    plt.text(x+0.4, text_y_loc, 'epoch '+ str(int(np.floor(x))), fontsize=8, rotation=90, color=color)
    

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.subplots_adjust(right=0.7)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(save_figure_directory + 'losses.png')
plt.show()