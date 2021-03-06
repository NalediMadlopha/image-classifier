import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from collections import OrderedDict
from utils import *


args = training_args()

# Transforms for the training     
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
   ]) 
}

# Load the datasets with ImageFolder
image_datasets = {
    "train": datasets.ImageFolder(args.data_dir + "train", transform=data_transforms["train"]),
    "valid": datasets.ImageFolder(args.data_dir + "valid", transform=data_transforms["test"])
}

# Define the dataloaders, using the image dataset and the trainforms
dataloaders = {
    "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
    "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=64, shuffle=True)
}

# Define the pre-trained models
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'alexnet': alexnet, 'vgg16': vgg16}
model = models[args.arch]

def main():
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = get_classifier(args.arch)
    
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train(model, args.epochs, 40, device)

def get_classifier(arch):
    switcher = {
        "alexnet": nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(9216, args.hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout()),
            ('fc2', nn.Linear(args.hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ])),
        "vgg16": nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, args.hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout()),
            ('fc2', nn.Linear(args.hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    }
    
    return switcher.get(arch, None)
        
    
def train(model, epochs, print_every, device):
    print(" --- TRAINING BEGIN --- ")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    steps = 0
    running_loss = 0

    for e in range(epochs):
        model.train()
        for images, labels in dataloaders["train"]:
            steps += 1

            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(model, dataloaders["valid"], device, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(dataloaders["valid"])),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders["valid"])))

                running_loss = 0

                model.train()
    
    save_checkpoint(model, args.arch, optimizer, args.save_dir)
    print(" --- TRAINING COMPLETED --- ")
    
def validation(model, testloader, device, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


def save_checkpoint(model, arch, optimizer, save_directory) :
    checkpoint = {
        'arch': arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'features': model.features,
        'state_dict': model.state_dict(),
        'idx_to_class': image_datasets["train"].class_to_idx
    }

    torch.save(checkpoint, save_directory + '/check_point.pth')
    
def test_accuracy(test_loader, device):    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    
if __name__ == "__main__":
    main()