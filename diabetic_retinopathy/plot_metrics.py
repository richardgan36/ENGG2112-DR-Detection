import matplotlib.pyplot as plt


# Function to plot accuracy and loss
def plot_graph(epochs, acc, val_acc, loss, val_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(epochs, acc, 'b')
    ax1.plot(epochs, val_acc, 'r')
    ax1.set_title("Model Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Val'], loc='upper left')

    ax2.plot(epochs, loss, 'b')
    ax2.plot(epochs, val_loss, 'r')
    ax2.set_title("Model Loss")
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Val'], loc='upper left')

    plt.show()
