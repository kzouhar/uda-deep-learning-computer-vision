import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import argparse
import csv
import time

from smdebug import modes
from smdebug.pytorch import get_hook
import smdebug.pytorch as smd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion, device, hook):
    '''
    This function takes a model and a testing data loader and will get the test accuray/loss of the model
    '''
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    if hook:
        hook.set_mode(modes.EVAL)

    running_loss=0.0
    running_corrects=0.0

    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += float(loss.item() * inputs.size(0))
        running_corrects += float(torch.sum(preds == labels.data))

    total_loss = float(running_loss) // float(len(test_loader.dataset))
    total_acc = float(running_corrects) // float(len(test_loader.dataset))

    print(f"Testing Loss: {total_loss}")
    print(f"Testing Accuracy: {total_acc}")

<<<<<<< HEAD
def train(model, train_loader, criterion, optimizer, device, hook, epochs):
    '''
    This function takes a model and data loaders for training and will get train the model
    '''
=======
def train(model, train_loader, criterion, optimizer, device, hook):
    '''
    This function takes a model and data loaders for training and will get train the model
    '''
    epochs=5

>>>>>>> 02f4a8002fa620f6f21cb2790dd76dcf6c4f7f19
    for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            model.train()
            hook.set_mode(modes.TRAIN)

            running_loss = 0.0
            running_corrects = 0.0
            running_samples=0

            total_samples_in_phase = len(train_loader.dataset)

            for inputs, labels in train_loader:
<<<<<<< HEAD
                inputs = inputs.to(device)
                labels = labels.to(device)
=======
                inputs=inputs.to(device)
                labels=labels.to(device)
>>>>>>> 02f4a8002fa620f6f21cb2790dd76dcf6c4f7f19
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += float(loss.item() * inputs.size(0))
                running_corrects += float(torch.sum(preds == labels.data))
                running_samples += len(inputs)

                accuracy = float(running_corrects)/float(running_samples)
                print("Epoch {}, Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                    epoch,
                    running_samples,
                    total_samples_in_phase,
                    100.0 * (float(running_samples) / float(total_samples_in_phase)),
                    loss.item(),
                    running_corrects,
                    running_samples,
                    100.0*accuracy,
                    ))

            epoch_loss = float(running_loss) // float(running_samples)
            epoch_acc = float(running_corrects) // float(running_samples)

<<<<<<< HEAD
            print('loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))
=======
            print('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(epoch_loss, epoch_acc))
>>>>>>> 02f4a8002fa620f6f21cb2790dd76dcf6c4f7f19

    return model


def create_pretrained_model():
    '''
    Create pretrained resnet50 model
    When creating our model we need to freeze all the convolutional layers which we do by their requires_grad() attribute to False.
    We also need to add a fully connected layer on top of it which we do use the Sequential API.
    '''
    model = models.resnet50(pretrained=True, progress=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 133))
    return model


def create_data_loaders(data, batch_size):
    # Modernized data loaders to skip downloading dataset every time
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    combined_dataset = ConcatDataset([train_data, validation_data])

    train_data_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader

def main(args):
    print(f'Hyperparameters: LR: {args.lr}, Batch Size: {args.batch_size}')
    print(f'Database Path: {args.data_path}')

    '''
    Create data loaders
    '''
    train_loader, test_loader=create_data_loaders(args.data_path, args.batch_size)

    '''
    Initialize pretrained model
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model=create_pretrained_model()
    model.to(device)

    '''
    Create loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    '''
    Create debug hook
    '''
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(criterion)

    '''
    Call the train function to start training model
    '''
    print("Starting Model Training")
    epochs = 5
    model = train(model, train_loader, criterion, optimizer, device, hook, epochs)
    
    '''
    Test the model to see its accuracy
    '''
    print("Testing Model")
    test(model, test_loader, criterion, device, hook)

    '''
    Save the trained model
    '''
    print("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    All the hyperparameters needed to use to train your model.
    '''
    # Training settings
    parser = argparse.ArgumentParser(description="Udacity AWS ML project 3 - Model training with debug")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument('--data_path', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args = parser.parse_args()

    main(args)
