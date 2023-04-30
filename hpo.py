import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset

import argparse

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader, criterion, device):
    '''
    This function takes a model and a testing data loader and will get the test accuray/loss of the model
    '''
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += float(loss.item() * inputs.size(0))
        running_corrects += float(torch.sum(preds == labels.data))

    total_loss = float(running_loss) // float(len(test_loader.dataset))
    total_acc = float(running_corrects) // float(len(test_loader.dataset))

    print(f"Testing Loss: {total_loss}")
    print(f"Testing Accuracy: {total_acc}")


def train(model, train_loader, criterion, optimizer, device) :
    '''
    This function takes a model and data loaders for training and will get train the model
    '''
    epochs = 5

    for epoch in range(epochs):

            print(f"Epoch: {epoch}")
            model.train()

            running_loss = 0.0
            running_corrects = 0.0
            running_samples = 0

            total_samples_in_phase = len(train_loader.dataset)

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)

                running_loss += float(loss.item() * inputs.size(0))
                running_corrects += float(torch.sum(preds == labels.data))
                running_samples += len(inputs)

                accuracy = float(running_corrects) / float(running_samples)

                print("Epoch {}, Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                    epoch,
                    running_samples,
                    total_samples_in_phase,
                    100.0 * (float(running_samples) / float(total_samples_in_phase)),
                    loss.item(),
                    running_corrects,
                    running_samples,
                    100.0 * accuracy,
                ))

            epoch_loss = float(running_loss) // float(running_samples)
            epoch_acc = float(running_corrects) // float(running_samples)

            print('loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))

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

    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')

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
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)



    return train_data_loader, test_data_loader


def main(args):

    print(f'Hyperparameters: LR: {args.lr}, Batch Size: {args.batch_size}')
    print(f'Database Path: {args.data_path}')

    '''
    Create data loaders
    '''
    train_loader, test_loader = create_data_loaders(args.data_path, args.batch_size)

    '''
    Initialize pretrained model
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model = create_pretrained_model()
    model.to(device)

    '''
    Create loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    '''
    Call the train function to start training model
    '''
    print("Starting Model Training")
    model = train(model, train_loader, criterion, optimizer, device)

    '''
    Test the model to see its accuracy
    '''
    print("Testing Model")
    test(model, test_loader, criterion, device)

    '''
    Save the trained model
    '''
    print("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == '__main__':
    '''
    All the hyperparameters needed to use to train your model.
    '''
    # Training settings
    parser = argparse.ArgumentParser(description="Udacity Udacity - Deep Learning : Computer Vision")
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
    print(args)

    main(args)
