"""
DISCLAIMER
I is not recommended to copy this code to avoid failing exams.
A check for plagiarism and task defense will be performed for 
each student! Uploaded for informational purposes only.
"""
"""
Convolutional neural network to recognize symbols using a MNIST-like datasets
Supported datasets: MNIST, EMNIST, KMNIST, QMNIST
Avg accuracy or 10 training epochs: 0.92
"""

print("==========================================================")
print("   CNN to recognize symbols using a MNIST-like datasets   ")
print("==========================================================")
print()
print("Loading libraries...")

from time import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Prepare for dataset loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Optimal tested value. Setting lower value may slow down training process
batch_size = 128

# Select dataset, uncomment necessary. Do not to change values in a whole file
# letters contains 27 different labels (a-z and unclassified) (0-26)
# digits contains 10 different labels (0-9)
split = "letters"
# split = "digits"

# Load dataset
print("Loading datasets...")
train_set = torchvision.datasets.EMNIST('./emnist', download=True, train=True, split=split, transform=transform)
train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = torchvision.datasets.EMNIST('./emnist', download=True, train=False, split=split, transform=transform)
test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


# Also MNIST is supported
# Uncomment the following code to use MNIST. Do not forget to comment 4 lines above
# Mnist contains 10 different labels (0-9)
# train_set = torchvision.datasets.MNIST('./emnist', download=True, train=True, transform=transform)
# train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_set = torchvision.datasets.MNIST('./emnist', download=True, train=False, transform=transform)
# test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


def train(mod, path):
    print("Neural network is training... It may take a while")
    # Load training dataset
    images, labels = next(iter(train_set_loader))

    # Reshape images to 1D to pass it to the NN input layers
    # images = images.view(images.shape[0], -1)

    probabilities = mod(images) # log probabilities

    # Set the loss function
    loss = nn.CrossEntropyLoss()
    p_output = loss(probabilities, labels)

    # Pass the backward to set the primary weights.
    # Weights are set to "None" by default
    p_output.backward()

    # Set an optimizer with default settings (torch documentation)
    optimizer = optim.SGD(mod.parameters(), lr=0.003, momentum=0.9)

    # Use it to calculate the training time
    time0 = time()

    # Setting lower value can be result of model inaccuracy.
    # Setting too high value may impact processing time and NN may be overtrained. It also can impact model accuracy.
    # Setting 10 is pretty enough to get good results. Datasets with larger amount of classes might require higher value.

    # Setting higher value just put my computer into a black screen and lagging task manager where I can't see how much
    # memory this program is used :( (I's a beautiful Intel UHD graphics that uses RAM instead of video memory because
    # it does not have own video memory)
    epochs = 10

    # Run training
    for e in range(epochs):
        running_loss = 0

        for images, labels in train_set_loader:
            # reshape images to 1D
            # images = images.view(images.shape[0], -1)

            # Zero gradients before backwarding
            optimizer.zero_grad()

            p_input = mod(images)

            # Calculate the loss
            p_output = loss(p_input, labels)

            # Backpropagation
            p_output.backward()

            # And optimize its weights here
            optimizer.step()

            # Calculate the whole loss
            running_loss += p_output.item()

        print("Epoch {}/{} - Training loss: {}".format(e + 1, epochs, running_loss / len(train_set_loader)))

    print("Training completed")
    print("Time: {}m {}s".format(int((time() - time0) / 60), int((time() - time0) % 60)))

    # Save the model for future use
    torch.save(mod, path)


def view(img, ps):
    ps = np.array(ps).squeeze()

    # Uncomment necessary
    size = 27 # EMNIST-LETTERS
    # size = 10 # EMNIST-DIGITS, MNIST

    fig, (ax1, ax2) = plt.subplots(figsize=(9, 6), ncols=2) # Set different values for figsize if displayed elements are too small

    # Show grayscaled image from dataset.
    # Remove transpose() if you are using MNIST.
    # Only EMNIST requires transpose() to show images correctly.
    ax1.imshow(img.resize_(1, 28, 28).numpy().transpose().squeeze(), cmap='gray_r')
    ax1.axis('off')

    # Show probability graph
    ax2.barh(np.arange(size), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(size))
    l = list()

    # Uncomment necessary
    l.extend("?ABCDEFGHIJKLMNOPQRSTUVWXYZ") # EMNIST-LETTERS
    # l.extend("0123456789") # EMNIST-DIGITS, MNIST

    ax2.set_yticklabels(l)
    ax2.set_title('Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


def test(path):
    print("Running random test...")
    try:
        mod = torch.load(path)

        images, labels = next(iter(test_set_loader))

        with torch.no_grad():
            logps = mod(images)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])

        l = list()

        # Uncomment necessary
        l.extend("?ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # EMNIST-LETTERS
        # l.extend("0123456789") # EMNIST-DIGITS, MNIST

        print("Predicted Symbol: " + l[probab.index(max(probab))] + ", accuracy:", max(probab))
        view(images[0].view(1, 28, 28), probab)
    except FileNotFoundError:
        print("Could not load model. Have you trained NN first?")


def test_all(path):
    print("Testing model accuracy... It may take a while")
    try:
        mod = torch.load(path)
        correct_count, all_count = 0, 0
        for images, labels in test_set_loader:
            for i in range(len(labels)):
                with torch.no_grad():
                    logps = mod(images)

                ps = torch.exp(logps)
                probab = list(ps.numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if true_label == pred_label:
                    correct_count += 1
                all_count += 1

        print("Number Of Images Tested =", all_count)
        print("Model Accuracy =", (correct_count / all_count))
    except FileNotFoundError:
        print("Could not load model. Have you trained NN first?")


if __name__ == "__main__":
    # Number of digits available (0-9)
    # Output layer
    output_size = 27 # EMNIST-LETTERS
    # output_size = 10 # EMNIST-DIGITS, MNIST

    # Set the NN
    print("Initializing CNN...")

    model = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding = 1),
                          nn.ReLU(),
                          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride = 1, padding = 1),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                          nn.Flatten(),
                          nn.Linear(in_features=9216, out_features=128),
                          nn.ReLU(),
                          nn.Linear(in_features=128, out_features=output_size),
                          nn.LogSoftmax(dim=1))

    # Set output path to the trained model
    model_path = './emnist-letters.pt'

    # Run
    # train(model, model_path) # It can be commented to speed up execution if you have already saved trained model
    test(model_path)
    test_all(model_path)
