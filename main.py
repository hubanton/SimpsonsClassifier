import os
from glob import glob
import torch.utils.data
import torch as torch
from PIL import Image
import numpy as np
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet18, convnext_tiny

# ----- EXERCISE 1 ----------------------------------------------------------------------------------------------------

# SET DEVICE TO GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SET GLOBAL SEED FOR DETERMINISM
torch.manual_seed(0)

# EXERCISE 1 a)

# The image path (in the same folder as pycharm project)
dataset_path = "imgs"

# Dict for mapping labels to indices (NN works with the indices)
name_dict = {}

train_x = []
train_y = []
val_x = []
val_y = []

# All folders (classes) containing the images
subdirectories = sorted(glob(os.path.join(dataset_path, "*")))

# Iterate through all folders and create the respective data partitions in the process
for subdir, index in zip(subdirectories, range(len(subdirectories))):
    name_dict[index] = subdir.split('\\')[1]

    image_list = sorted(glob(os.path.join(subdir, "*.jpg")))

    length = len(image_list)

    split = int(length * 0.6)

    # Fill train x and y into a list
    for i in range(split):
        train_x.append(image_list[i])
        train_y.append(index)
    # Fill validation x and y into a list
    for i in range(split, length):
        val_x.append(image_list[i])
        val_y.append(index)


# EXERCISE 1 b)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.num_examples = len(self.images)
        self.trans = transforms.Compose([transforms.Resize(128)])

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = Image.open(img_name).convert('RGB')

        img = np.asarray(img)

        label = np.asaray(self.labels[idx])

        img = img / 255.0
        img = img * 2
        img = img - 1
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)

        img = torch.from_numpy(img.copy()).float()
        img = self.trans(img)
        label = torch.from_numpy(label.copy()).long()
        return img, label

    def __len__(self):
        return self.num_examples


# Create a wrapper around our train data (Only necessary for train since we always use the entire val set)
train_Dataset = CustomDataset(train_x, train_y)


# CHANGES MADE TO DATALOADER for more deterministic behaviour amongst workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    torch.random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

# Create DataLoader with the new wrapper (and batch_size=16)
train_dataloader = DataLoader(train_Dataset, batch_size=64, shuffle=True, drop_last=True,
                              worker_init_fn=seed_worker,
                              generator=g)

# Create the iter...
data_loader_iter = iter(train_dataloader)


# EXERCISE 1 c)
def train_step(module, criterion, optimizer, X, y, gradScaler):
    # Switch to training mode
    module.train()

    # Set gradients to zero before new training step
    optimizer.zero_grad()

    # EXERCISE 1 e) Extend the train step to work with mixed precision training
    with autocast(device_type=device, dtype=torch.float16):
        # Model prediction
        output = module(X)
        # Calculate loss of prediction on train set
        loss = criterion(output, y)

    # Update the weights (now with mixed precision)
    gradScaler.scale(loss).backward()
    gradScaler.step(optimizer)

    return loss


def train_and_evaluate(module, criterion, optimizer, train_loader,
                       epochs, x_val, y_val, gradScaler, use_last_module=False):
    # EXERCISE 1 g)
    # In case an old model exists and we want to reuse it, we can load the old weights and optimizer from memory
    if use_last_module and os.path.exists('default_checkpoint.pth'):
        last_checkpoint = torch.load('default_checkpoint.pth')
        if last_checkpoint is not None:
            module.load_state_dict(last_checkpoint['model'])
            optimizer.load_state_dict(last_checkpoint['optimizer'])

    # Accuracy of each epoch
    epoch_accuracy_list = []

    for epoch in range(epochs):
        # One training epoch
        for k, (batch_x, batch_y) in enumerate(train_loader):
            module.train()
            train_step(module, criterion, optimizer, batch_x, batch_y, gradScaler)

    # Switch to evaluation and calculate accuracy
    module.eval()

    # transform the output to labels (we want to pick the label where output is the largest)
    output = module(x_val)
    output = torch.argmax(output, dim=1)

    # Calculate accuracy for all classes and then take the mean of that (With confusion matrix)
    matrix = confusion_matrix(output, y_val)
    accuracy = matrix.diagonal() / matrix.sum(axis=0)
    accuracy = accuracy.mean()

    # If this epoch is the best (regarding accuracy on test) so far, save both weights and state inside checkpoint.pth
    # ONLY UPDATE MODEL IN MEMORY IF THERE WERE IMPROVEMENTS
    if accuracy > max(epoch_accuracy_list) or len(epoch_accuracy_list) == 0:
        checkpoint = {
            'model': module.state_dict(),
            'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, 'best_epoch_checkpoint.pth')

    # Save current state for each epoch (regardless of improvement or not [In case the server crashes])
    checkpoint = {
        'model': module.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'default_checkpoint.pth')

    # Store epoch acc in list
    epoch_accuracy_list.append(accuracy)

    return epoch_accuracy_list


# EXERCISE 1 d)
# Load the empty resnet18 model
res_model = resnet18()
# Change the final layer to be a liner with 512 inputs and 37 outputs (Number of simpsons characters)
res_model.fc = torch.nn.Linear(512, len(name_dict), bias=True)

# Load the empty convnext_tiny model
convnext_model = convnext_tiny()
# Also replace the last layer with a linear layer with 37 outputs
convnext_model.classifier = torch.nn.Linear(512, len(name_dict), bias=True)

# EXERCISE 1 e) (See the further adjustments in the train_step function)
scaler = GradScaler()

# EXERCISE 1 f) (Further Changes can be found throughout the code)

# Tell torch to avoid any non-deterministic algorithms
torch.use_deterministic_algorithms(True)

# ----- EXERCISE 2 ----------------------------------------------------------------------------------------------------

adamOptimizer = torch.optim.Adam(res_model.parameters(), lr=0.0001)

crossEntropy = torch.nn.CrossEntropyLoss()

trainEpochs = 30

res_model_training_accuracy = train_and_evaluate(module=res_model, criterion=crossEntropy, optimizer=adamOptimizer,
                                                 train_loader=train_dataloader, epochs=trainEpochs,
                                                 x_val=val_x, y_val=val_y, gradScaler=scaler,
                                                 use_last_module=False)

convnext_model = train_and_evaluate(module=convnext_model, criterion=crossEntropy, optimizer=adamOptimizer,
                                    train_loader=train_dataloader, epochs=trainEpochs,
                                    x_val=val_x, y_val=val_y, gradScaler=scaler,
                                    use_last_module=False)

