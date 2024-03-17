import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def extract_data(line, current_epoch):
    epoch_pattern = r'epoch: (\d+)'
    loss_pattern = r'tot ([\d.]+) \| hm ([\d.]+) \| wh ([\d.]+) \| reg ([\d.]+) \| ltrb_amodal ([\d.]+) \| tracking ([\d.]+) \| time ([\d.]+)'
    
    # Check if the line contains the 'epoch' keyword
    epoch_match = re.search(epoch_pattern, line)
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
        is_validation = False
    else:
        is_validation = True
    
    # Extract the loss values
    loss_match = re.search(loss_pattern, line)
    if loss_match:
        loss_data = {
            'epoch': current_epoch,
            'tot': float(loss_match.group(1)),
            'hm': float(loss_match.group(2)),
            'wh': float(loss_match.group(3)),
            'reg': float(loss_match.group(4)),
            'ltrb_amodal': float(loss_match.group(5)),
            'tracking': float(loss_match.group(6)),
            'time': float(loss_match.group(7)),
        }
        return loss_data, is_validation
    return None, None

def read_and_parse_log(file_path):
    training_losses = []
    validation_losses = []
    current_epoch = 0  # Initialize current_epoch
    with open(file_path, 'r') as file:
        for line in file:
            loss_data, is_validation = extract_data(line, current_epoch)
            if loss_data:
                if is_validation:
                    validation_losses.append(loss_data)
                else:
                    training_losses.append(loss_data)
                    current_epoch = loss_data['epoch']  # Update current_epoch
    return training_losses, validation_losses

def plot_losses(training_losses, validation_losses, save_path):
    df_training = pd.DataFrame(training_losses)
    df_validation = pd.DataFrame(validation_losses)
    
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'black', 'purple']
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, metric in enumerate(['tot', 'hm', 'wh', 'reg', 'ltrb_amodal', 'tracking', 'time']):
        ax = axes[i]
        df_training.plot(x='epoch', y=metric, ax=ax, label='Training', color=colors[i])
        df_validation.plot(x='epoch', y=metric, ax=ax, label='Validation', linestyle='--', color=colors[i])
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True)  # Enable grid

    plt.tight_layout()
    plt.savefig(save_path)

def main():
    log_file_path = "./log.txt"
    now = datetime.now() 
    save_path = f'{now}.png'

    training_losses, validation_losses = read_and_parse_log(log_file_path)
    plot_losses(training_losses, validation_losses, save_path)

if __name__ == '__main__':
    main()
