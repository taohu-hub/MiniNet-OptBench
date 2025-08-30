import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch import nn, optim
from tqdm import tqdm
import time
import numpy as np

# Define the neural network models (NanoGPT, MLP, etc.)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.Linear(output_dim, 10)  # for CIFAR10

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the optimizer test function
def run_optimizer_test(optimizer_name, model, data_loader, criterion, device):
    optimizer = get_optimizer(optimizer_name, model)
    
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for inputs, targets in tqdm(data_loader, desc=f"Training with {optimizer_name}"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        _, predicted = outputs.max(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    duration = time.time() - start_time
    accuracy = 100 * correct / total
    avg_loss = epoch_loss / len(data_loader)
    return accuracy, avg_loss, duration

def get_optimizer(optimizer_name, model):
    if optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name == 'Muon':
        # Add Muon optimizer code here
        pass
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")

def load_data(task):
    if task == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        return trainloader
    elif task == 'Fashion-MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        return trainloader
    else:
        raise ValueError(f"Dataset {task} not recognized.")

def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the model and load data for the specified task
    input_dim = 28 * 28  # Example for Fashion-MNIST, change for other tasks
    hidden_dim = 512
    output_dim = 256
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)

    # Load data
    data_loader = load_data(args.task)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Run optimizer tests
    accuracies = []
    losses = []
    durations = []

    for optimizer_name in args.optimizers:
        accuracy, loss, duration = run_optimizer_test(optimizer_name, model, data_loader, criterion, device)
        accuracies.append(accuracy)
        losses.append(loss)
        durations.append(duration)

        print(f"Optimizer: {optimizer_name}")
        print(f"Accuracy: {accuracy:.2f}% | Loss: {loss:.4f} | Time: {duration:.2f} sec")

    # Save or log results if necessary
    with open('benchmark_results.txt', 'w') as f:
        for optimizer_name, accuracy, loss, duration in zip(args.optimizers, accuracies, losses, durations):
            f.write(f"{optimizer_name}: Accuracy = {accuracy:.2f}% | Loss = {loss:.4f} | Time = {duration:.2f} sec\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Neural Network Optimizer Benchmark")
    parser.add_argument('--task', type=str, choices=['CIFAR10', 'Fashion-MNIST'], required=True, help="Task to benchmark (CIFAR10/Fashion-MNIST)")
    parser.add_argument('--optimizers', nargs='+', default=['SGD', 'Adam'], help="List of optimizers to test")

    args = parser.parse_args()
    main(args)
