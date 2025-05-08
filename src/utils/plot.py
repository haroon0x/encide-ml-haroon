import matplotlib.pyplot as plt

def plot_training_history(history, fine_tune_history=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs_initial = range(1, len(history.history['accuracy']) + 1)
    ax1.plot(epochs_initial, history.history['accuracy'], 'b-', label='Training Accuracy')
    ax1.plot(epochs_initial, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
    
    if fine_tune_history:
        epochs_fine = range(len(history.history['accuracy']) + 1, 
                          len(history.history['accuracy']) + len(fine_tune_history.history['accuracy']) + 1)
        ax1.plot(epochs_fine, fine_tune_history.history['accuracy'], 'g-', label='Fine-tuning Training Accuracy')
        ax1.plot(epochs_fine, fine_tune_history.history['val_accuracy'], 'm-', label='Fine-tuning Validation Accuracy')
    
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs_initial, history.history['loss'], 'b-', label='Training Loss')
    ax2.plot(epochs_initial, history.history['val_loss'], 'r-', label='Validation Loss')
    
    if fine_tune_history:
        epochs_fine = range(len(history.history['loss']) + 1, 
                          len(history.history['loss']) + len(fine_tune_history.history['loss']) + 1)
        ax2.plot(epochs_fine, fine_tune_history.history['loss'], 'g-', label='Fine-tuning Training Loss')
        ax2.plot(epochs_fine, fine_tune_history.history['val_loss'], 'm-', label='Fine-tuning Validation Loss')
    
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()