import matplotlib.pyplot as plt
import json
import os


def plot_training_results(file_path, save_name='training_plot.png'):
    #Loading the data
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    #Extracting training and validation logs
    train_steps = [item['step'] for item in data['training_losses']]
    train_values = [item['loss'] for item in data['training_losses']]

    val_steps = [item['step'] for item in data['validation_losses']]
    val_values = [item['loss'] for item in data['validation_losses']]

    #Calculate smoothed trend
    window = 10
    smoothed_train = [
        sum(train_values[max(0, i - window):i + 1]) / len(train_values[max(0, i - window):i + 1])
        for i in range(len(train_values))
    ]


    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_values, color='skyblue', alpha=0.3, label='Train Loss (Raw)')
    plt.plot(train_steps, smoothed_train, color='dodgerblue', linewidth=2, label='Train Loss (Smoothed)')
    plt.plot(val_steps, val_values, color='crimson', marker='o', markersize=4, label='Validation Loss')

    # 5. Formatting
    plt.title(f"Training Progress: {data['config']['model']}", fontsize=12)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    #Best validation
    best_val = data.get('best_val_loss')
    if best_val and best_val in val_values:
        plt.annotate(f'Best Val: {best_val}',
                     xy=(val_steps[val_values.index(best_val)], best_val),
                     xytext=(20, 20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()

    #Saving the plot
    plt.savefig(save_name, dpi=300)
    print(f"Plot successfully saved as {save_name}")


    plt.close()


# Run the function
plot_training_results('loss_log.json')