
import torchvision, time, os, copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # ImageNet models were trained on 224x224 images
        transforms.RandomHorizontalFlip(), # flip horizontally 50% of the time - increases train set variability
        transforms.ToTensor(), # convert it to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet models expect this norm
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# data_dir = 'data/hymenoptera_data'
data_dir = input("Please input hymenoptera data path :")
# Create train and validation datasets and loaders
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
    for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Helper function for displaying images
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Un-normalize the images
    inp = std * inp + mean
    # Clip just in case
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show ()


# Get a batch of training data
#inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])
# training

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    epoch_time = [] # we'll keep track of the time needed for each epoch

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # Add the epoch time
        t_epoch = time.time() - epoch_start
        epoch_time.append(t_epoch)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_time


# Download a pre-trained ResNet18 model and freeze its weights
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# # Replace the final fully connected layer
# # Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
# # Send the model to the GPU
model = model.to(device)
# # Set the loss function
criterion = nn.CrossEntropyLoss()

# Observe that only the parameters of the final layer are being optimized
optimizer_conv = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
model, epoch_time = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=2)

#---------------------------------------------------------------------
# import cv2
# from uuid import uuid4


# def take_picture():

#     cap = cv2.VideoCapture(0)
#     if cap.isOpened():
#         cv2.namedWindow("Test", cv2.WINDOW_AUTOSIZE)
#         k = 0
#         while cv2.getWindowProperty("Test", 0) >= 0:
#             _, img = cap.read()
#             cv2.imshow("Test", img)
#             keyCode = cv2.waitKey(30) & 0xFF
#             # Stop the program on the ESC key
#             if keyCode == 27:
#                 break
#             if keyCode == 32:
#                 path = os.getcwd()+f"\\tl_data\\img{k}.png"
#                 cv2.imwrite(path,img)
#                 start_time = time.time()
#                 test_img = Image.open(path)
#                 test_image = data_transforms['val'](test_img).unsqueeze(0)
#                 out = model(test_image)
#                 _, pred = torch.max(out, 1)
#                 pred = int(pred)
#                 end_time = time.time()
#                 if pred == 0:
#                     print(f"Prédiction img{k}: Fourmi")
#                 else:
#                     print(f"Prédiction img{k}: Abeille")
#                 print(f"prediction time {end_time-start_time:.3f} s")
#                 k += 1
#         cap.release()
#         cv2.destroyAllWindows()
#     else:
#         print("Unable to open camera")
    
# take_picture()
#-------------------------------------------------------------------

f = []
mypath = os.getcwd()+'\\tl_data'
for (dirpath, dirnames, filenames) in os.walk(mypath):
    f.extend(filenames)

f = [fn for fn in f if 'png' in fn]

for image_path in f:

    start_time = time.time()
    
    image = Image.open(mypath+'/'+image_path)

    # # Now apply the transformation
    image = data_transforms['val'](image).unsqueeze(0)

    out = model(image)
    _, pred = torch.max(out, 1)
    pred = int(pred)
    end_time = time.time()
    if pred == 0:
        print(f"Prédiction {image_path}: Fourmi")
    else:
        print(f"Prédiction {image_path}: Abeille")
    print(f"Prediction time {end_time-start_time:.3f} s")

    