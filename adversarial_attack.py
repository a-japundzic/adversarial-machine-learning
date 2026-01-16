import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from VGG11 import VGG11
import os

# Create folder for images
if not os.path.exists('e3_results'):
    os.mkdir('e3_results')

# Method to train the model (reformatted it this way to make parts c and d easier)
def train_model(train_loader, model, device, loss_function, optimizer):
    # Train the model (following the training loop and per epoch activity from the pytorch documentation)
    # Set the model to training mode and train one epoch
    model.train(True)

    # Total loss for the epoch
    running_loss = 0
    # Total number of labels in the epoch
    total_labels = 0
    # Total number of correct predictions in the epoch
    correct = 0
    for (images, labels) in train_loader:
        # Move the images and labels to the GPU
        images = images.to(device)
        labels = labels.to(device)

        # zero the gradients
        optimizer.zero_grad()

        # Make predictions
        outputs = model(images)

        # Calculate the loss and its gradients (as discussed in class)
        loss = loss_function(outputs, labels)
        loss.backward()

        # Adjust the learning rate
        optimizer.step()

        # Track statistics
        # Have to multiply by the batch size to get the total loss for the batch
        running_loss += loss.item() * images.size(0)
        # Predict the class with the highest probability (10 classes for MINST, each correspondding to a number 0-9)
        predicted = torch.argmax(outputs, dim=1)
        # Add up the total number of labels in the batch
        total_labels += labels.size(0)
        # Add up the total number of correct predictions in the batch
        correct += (predicted == labels).sum().item()

    # Calculate the average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct / total_labels
    print(f"  Train Loss: {epoch_loss:.4f} || Train Accuracy: {epoch_accuracy:.4f}")


    return epoch_loss, epoch_accuracy

def train_model_attack(train_loader, model, device, loss_function, optimizer, eps, attack_type):
    # Set the model ton trainig mode
    model.train(True)

    # Total loss for the epoch
    running_loss = 0
    # Total number of labels in the epoch
    total_labels = 0
    # Total number of correct predictions in the epoch
    correct = 0
    for (images, labels) in train_loader:
        # Move the images and labels to the GPU
        images = images.to(device)
        labels = labels.to(device)

        # Alter the images using the attack type
        if attack_type == 'fgsm':
            # We need to find the gradients so set the feature for the tensors
            images.requires_grad = True

            # Make predictions (forward pass)
            outputs = model(images)

            # Zero the gradients
            model.zero_grad()

            # Calculate the loss and its gradients (as discussed in class)
            loss = loss_function(outputs, labels)
            # Calculate the gradients (backpropagation)
            loss.backward()

            # Collect the gradients
            data_grad = images.grad.data

        
            x_mod = fgsm(images, eps, data_grad)
        # Otherwise use pgd
        else:
            x_mod = pgd(images, model, loss_function, labels, eps, 0.02, 5)

        # Make predictions on the modified images
        outputs = model(x_mod)

        # Zero the gradients
        model.zero_grad()

        # Calcucalte the loss on the modified images
        loss = loss_function(outputs, labels)
        # Calculate the gradients (backpropagation)
        loss.backward()

        # Adjust the learning rate
        optimizer.step()

        # Track statistics
        # Have to multiply by the batch size to get the total loss for the batch
        running_loss += loss.item() * images.size(0)
        # Predict the class with the highest probability (10 classes for MINST, each correspondding to a number 0-9)
        predicted = torch.argmax(outputs, dim=1)
        # Add up the total number of labels in the batch
        total_labels += labels.size(0)
        # Add up the total number of correct predictions in the batch
        correct += (predicted == labels).sum().item()

    # Calculate the average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct / total_labels
    print(f"  Train Loss {attack_type}: {epoch_loss:.4f} || Train Accuracy {attack_type}: {epoch_accuracy:.4f}")

    return epoch_loss, epoch_accuracy

# Method to test the model (reformatted it this way to make parts c and d easier)
def test_model(test_loader, model, device, loss_function):
    # Set the model to evaluation mode to test
    model.eval()

    # Total loss for the epoch
    running_loss = 0
    # Total number of labels in the epoch
    total_labels = 0
    # Total number of correct predictions in the epoch
    correct = 0
    # Disable the gradient calculation as we are no longer optimizing
    # (i.e. no need for backpropagation)
    # This should speed up the testing process
    with torch.no_grad():
        for (images, labels) in test_loader:
            # Move the images and labels to the GPU (or to CPU if no GPU is available)
            images = images.to(device)
            labels = labels.to(device)
            
            # Make predictions
            outputs = model(images)

            # Calculate the loss (no need for gradients as we are not optimizing)
            loss = loss_function(outputs, labels)
            # Again have to multiply by the batch size to get the total loss for the batch
            running_loss += loss.item() * images.size(0)
            # Predict the class with the highest probability
            predicted = torch.argmax(outputs, dim=1)
            # Add up the toal number of labels in the batch
            total_labels += labels.size(0)
            # Add up the toal number of correct predictions in the batch
            correct += (predicted == labels).sum().item()

    # Calculate the average loss and accuracy for the epoch
    epoch_test_loss = running_loss / len(test_loader.dataset)
    epoch_test_acc = correct / total_labels
    print(f"  Test Loss: {epoch_test_loss:.4f}  || Test Accuracy: {epoch_test_acc:.4f}")

    return epoch_test_loss, epoch_test_acc

def test_model_attack(test_loader, model, device, loss_function, eps, attack_type):
    # Set the model to evaluation mode to test
    model.eval()

    # Total loss for the epoch
    running_loss = 0
    # Total number of labels in the epoch
    total_labels = 0
    # Total number of correct predictions in the epoch
    correct = 0
    # Using gradient this time to find delta

    # Image examples
    sample = []
    for (images, labels) in test_loader:
        # Move the images and labels to the GPU (or to CPU if no GPU is available)
        images = images.to(device)
        labels = labels.to(device)

        # Alter the images using the attack type
        if attack_type == 'fgsm':
            # We need to find the gradients so set the feature for the tensors
            images.requires_grad = True

            # Make predictions (forward pass)
            outputs = model(images)

            # Zero the gradients
            model.zero_grad()

            # Calculate the loss and its gradients (as discussed in class)
            loss = loss_function(outputs, labels)
            # Calculate the gradients (backpropagation)
            loss.backward()

            # Collect the gradients
            data_grad = images.grad.data

            x_mod = fgsm(images, eps, data_grad)
        # Otherwise use pgd
        else:
            x_mod = pgd(images, model, loss_function, labels, eps, 0.02, 5)

        # Make predictions on the modified images
        outputs = model(x_mod)

        # Predict the class with the highest probability
        predicted = torch.argmax(outputs, dim=1)

        # Append the modified image, label, and predicted label to the list
        for i in random.sample(range(len(images)), 5):
            # Append the modified image, label, and predicted label to the list
            sample.append((x_mod[i].detach().cpu(), labels[i].detach().cpu().item(), predicted[i].detach().cpu().item()))


        # Calculate the loss on the modified images
        loss = loss_function(outputs, labels)
        # Again have to multiply by the batch size to get the total loss for the batch
        running_loss += loss.item() * images.size(0)
        # Add up the toal number of labels in the batch
        total_labels += labels.size(0)
        # Add up the total number of correct predictions in the batch
        correct += (predicted == labels).sum().item()

    # Calculate the average loss and accuracy for the epoch
    epoch_test_loss = running_loss / len(test_loader.dataset)
    epoch_test_acc = correct / total_labels
    print(f"  Test Loss {attack_type}: {epoch_test_loss:.4f}  || Test Accuracy {attack_type}: {epoch_test_acc:.4f}")

    return epoch_test_loss, epoch_test_acc, sample
    
def fgsm(image, eps, data_grad):
    # Find delta_star following fgsm
    delta_star = eps * data_grad.sign()

    # Alter the imgage by adding delta_star
    x_mod = image + delta_star

    # Clip the image to be between 0 and 1 (keep it black and white)
    x_mod = torch.clamp(x_mod, 0, 1)

    # Return the modified image
    return x_mod

def pgd(image, model, loss_function, labels, eps, eta, iters):
    delta = torch.zeros_like(image)
    # Set original and modified images
    x = image.clone()
    x_mod = image.clone()


    for _ in range(iters):
        # Add to device
        x_mod = x_mod.to(device)
        x = x.to(device)

        # Need to find the gradient first
        x_mod.requires_grad = True

        # Do a forward pass
        outputs = model(x_mod)

        # Find the loss
        loss = loss_function(outputs, labels)

        # Backpropagation
        model.zero_grad()
        loss.backward()

        # Collect the gradients
        data_grad = x_mod.grad.data

        # Calculate the new delta
        delta = delta + eta * data_grad.sign()
        # Project the delta to be between -eps and eps
        delta = torch.clamp(delta, -eps, eps)

        # Alter the image by adding the delta
        x_mod = x + delta
        # Clip the image to be between 0 and 1 (keep it black and white)
        x_mod = torch.clamp(x_mod, 0, 1)

    # Return the modified image
    return x_mod

if __name__ == "__main__":
    # Variable to hold the number of epochs (in this case 5)
    EPOCH = 5
    
    # a)
    print("Basic: No generalization or regularization")
    # Transform the data
    transform_a = transforms.Compose([
        # 32 by 32 as mentioned in the assignment
        transforms.Resize((32, 32)),
        # Convert the image to a tensor (as required by PyTorch)
        transforms.ToTensor(),
        # Following what the paper does and subtracting the mean pixel value
        # I searched up the MNIST mean and found that it is 0.1307
        # transforms.Normalize(mean=[0.1307], std=[1.0])
    ])

    # Load the data (shuffle the training data for each epoch)
    train_dataset_a = datasets.MNIST(root='./data', train=True, download=True, transform=transform_a)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_a)

    # Wrap in DataLoader
    # Set batch size to 256 (followed what the paper did)
    train_loader_a = DataLoader(train_dataset_a, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Setup the model so that it is on the GPU (im lucky enough to have a 3080 for this assignment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG11().to(device)

    # Setup the loss function (assignment said to use cross entropy loss)
    loss_function = nn.CrossEntropyLoss()
    # Setup the optimizer (followed what the paper did)
    optimizer = optim.SGD(model.parameters(), weight_decay=5e-4, lr=0.01, momentum=0.9)

    # Lists to store metrics for plotting
    train_losses_a = []
    train_accuracies_a = []
    test_losses_a = []
    test_accuracies_a = []

    # Get a sample of the original images
    images, labels = next(iter(test_loader))
    images = images.cpu()
    labels = labels.cpu()

    # Plot 5 of the original images for comparison
    k = 5
    plt.figure(figsize=(10, 2))
    for j, i in enumerate(random.sample(range(len(images)), k)):
        plt.subplot(1, k, j + 1)
        # Squeeze the image to remove the channel dimension
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Label:{labels[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('e3_results/image_example_original.png')
    plt.close()

    # Train and test model for each epoch
    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}:")
        # Train and test!
        # epoch_train_loss, epoch_train_acc = train_model(train_loader_a, model, device, loss_function, optimizer)
        epoch_train_loss, epoch_train_acc = train_model_attack(train_loader_a, model, device, loss_function, optimizer, eps=0.2, attack_type='fgsm')
        # epoch_test_loss, epoch_test_acc = test_model(test_loader, model, device, loss_function)
        epoch_test_loss, epoch_test_acc, sample = test_model_attack(test_loader, model, device, loss_function, eps=0.2, attack_type='pgd')

        # Plot 5 of the modified images for comparison
        k = 5
        # Pick 5 random indices from the returned sample
        indices = random.sample(range(len(sample)), k)
        
        # Plot the 5 modified images
        plt.figure(figsize=(10, 2))
        for i, idx in enumerate(indices):
            img = sample[idx][0]
            true_lbl = sample[idx][1]
            pred_lbl = sample[idx][2]

            plt.subplot(1, k, i + 1)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f"Label:{true_lbl} Pred:{pred_lbl}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'e3_results/image_example_{epoch + 1}.png')
        plt.close()


        # Append the metrics to the lists for plotting
        train_losses_a.append(epoch_train_loss)
        train_accuracies_a.append(epoch_train_acc)
        # test_losses_a.append(epoch_test_loss)
        # test_accuracies_a.append(epoch_test_acc)
        test_losses_a.append(epoch_test_loss)
        test_accuracies_a.append(epoch_test_acc)

    # b)
    # Plot
    # Spent some time producing this cool graph, hope you enjoy it!
    epochs = range(1, EPOCH + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    
    axs[0, 0].plot(epochs, test_accuracies_a, 'b-o', label='Test Acc')
    axs[0, 0].set_title('Test Accuracy (Default) (higher is better)')
    axs[0, 0].set_ylabel('Accuracy (%)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(epochs, train_accuracies_a, 'g-o', label='Train Acc')
    axs[0, 1].set_title('Training Accuracy (EPS=0.2, ATTACK=PGD) (higher is better)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(epochs, test_losses_a, 'r-o', label='Test Loss')
    axs[1, 0].set_title('Test Loss (Default) (lower is better)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(epochs, train_losses_a, 'm-o', label='Train Loss')
    axs[1, 1].set_title('Training Loss (EPS=0.2, ATTACK=PGD) (lower is better)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
