import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
import torch.nn.functional as F

torch.backends.cudnn.enabled = False

import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, recall_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm


####### MAIN TUTORIAL: https://www.youtube.com/watch?v=tHL5STNJKag&ab_channel=RobMulla

# clasa care defineste dataset-ul
class SymbolsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


# functia care da resize la fiecare poza ca ne asiguram ca sunt la fel
transform = transforms.Compose([transforms.Resize((280, 280)), transforms.ToTensor()])

# incarcam imaginile si cream un dictionar care sa asocieze numarul (label-ul) imaginii cu numele clasei
data_dir = 'images'
dataset = SymbolsDataset(data_dir, transform=transform)
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
dataset_size = len(dataset)

# impartim in 60% train / 20% validation/ 20% test
main_data, test_data = random_split(dataset, [5892, 1474])
train_data, validation_data = random_split(main_data, [4418, 1474])

# incarcam dataseturile pentru a putea fi prelucrate in paralel
# shuffle decide daca datele sunt luate random, sau in ordine
# batch_size = cate imagini sunt luate intr-o iteratie
# imparte datasetul in grupuri pentru a fi prelucrate mai rapid
train_loaderData = DataLoader(train_data, batch_size=16, shuffle=True)
validation_loaderData = DataLoader(validation_data, batch_size=16, shuffle=True)
test_loaderData = DataLoader(test_data, batch_size=16, shuffle=True)


class PretrainedModel(nn.Module):
    def __init__(self):
        super(PretrainedModel, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # 6 clase de output, cream classifier-ul
        self.classifier = nn.Linear(enet_out_size, 6)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


# 2 conv, 4 layers, Relu
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1_input_size = 16 * 67 * 67
        self.fc1 = nn.Linear(self.fc1_input_size, 144)
        self.fc2 = nn.Linear(144, 72)
        self.fc3 = nn.Linear(72, 36)
        self.fc4 = nn.Linear(36, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# same but with sigmoid
class SigmoidModel(nn.Module):
    def __init__(self):
        super(SigmoidModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1_input_size = 16 * 67 * 67
        self.fc1 = nn.Linear(self.fc1_input_size, 144)
        self.fc2 = nn.Linear(144, 72)
        self.fc3 = nn.Linear(72, 36)
        self.fc4 = nn.Linear(36, 6)

    def forward(self, x):
        x = self.pool(F.sigmoid(self.conv1(x)))
        x = self.pool(F.sigmoid(self.conv2(x)))
        x = x.view(-1, self.fc1_input_size)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x


# same as first but with softmax
class SoftmaxModel(nn.Module):
    def __init__(self):
        super(SoftmaxModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1_input_size = 16 * 67 * 67
        self.fc1 = nn.Linear(self.fc1_input_size, 144)
        self.fc2 = nn.Linear(144, 72)
        self.fc3 = nn.Linear(72, 36)
        self.fc4 = nn.Linear(36, 6)

    def forward(self, x):
        x = self.pool(F.softmax(self.conv1(x), dim=1))
        x = self.pool(F.softmax(self.conv2(x), dim=1))
        x = x.view(-1, self.fc1_input_size)
        x = F.softmax(self.fc1(x), dim=1)
        x = F.softmax(self.fc2(x), dim=1)
        x = F.softmax(self.fc3(x), dim=1)
        x = self.fc4(x)
        return x


def predict_class(result_tensor):
    # softmax = torch.nn.functional.softmax(result_tensor, dim=1)
    predicted_class = torch.argmax(result_tensor, dim=1).item()
    return target_to_class[predicted_class]


def train():
    # definim device-ul pe care sa ruleze modelul (GPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # definesc numarul de epoci , o epoca e o iteratie prin tot datasetul de train
    num_epochs = 50
    train_losses, valid_losses = [], []

    model = BaseModel()
    model.to(device)

    # loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loaderData, desc='Training loop'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loaderData)
        train_losses.append(train_loss)

        # evaluam
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(validation_loaderData, desc="Validation loop"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
            valid_loss = running_loss / len(validation_loaderData)
            valid_losses.append(valid_loss)

        # printam statistica epocii
        print(f"\nEpoch {epoch + 1}/{num_epochs} - Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}\n ")
        # modificam scheduler-ul dupa fiecare epoca
        scheduler.step()

    # vizualizam
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.show()

    torch.save(model.state_dict(), 'LR_Scheduler.pth')


def showDatasetStats():
    print('dataset size: ' + str(dataset_size))

    # numar cate imagini sunt pe fiecare clasa
    class_counts = [0] * len(target_to_class)
    for _, label in dataset:
        class_counts[label] += 1

    class_names = [target_to_class[i] for i in range(len(target_to_class))]

    # histograma
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts)
    plt.xlabel('Classes')
    plt.ylabel('Number')
    plt.title('Histogram')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def showConfusionMatrix(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    classes = np.arange(len(cm))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           title="Confusion Matrix",
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = np.max(cm) / 2.
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            ax.text(j, i, format(cm[i][j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black")
    fig.tight_layout()

    plt.show()


def test():
    model = BaseModel()
    model.load_state_dict(torch.load("LR_Scheduler.pth"))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loaderData, desc="Testing..."):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted_classes = torch.argmax(outputs, dim=1)
            correct += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (correct / total_samples) * 100
    print(f'Accuracy: {accuracy:.2f}%')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    showConfusionMatrix(cm)

    # Recall
    recall = recall_score(all_labels, all_predictions, average=None)
    print("Recall for each class:")
    print(recall)


if __name__ == '__main__':
    # train()
    test()
    # showDatasetStats()