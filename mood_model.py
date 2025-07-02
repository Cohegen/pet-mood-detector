import torch 
from torch import nn
from torchvision import transforms, datasets
import os
import cv2
from PIL import Image

def get_classes(data_dir):
    classes = []
    for animal in os.listdir(data_dir):
        animal_path = os.path.join(data_dir, animal)
        if os.path.isdir(animal_path):
            for mood in os.listdir(animal_path):
                mood_path = os.path.join(animal_path, mood)
                if os.path.isdir(mood_path):
                    classes.append(f'{animal}_{mood}')
    return classes

data_dir = 'dataset'
class_names = get_classes(data_dir)
print('Classes:', class_names)

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Custom target_transform to combine animal and mood as class
def target_transform(target):
    return target

# Loading the dataset
full_dataset = datasets.ImageFolder(
    root=data_dir,
    transform=train_transform
)
print('ImageFolder classes:', full_dataset.classes)

class Mood(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.flatten = nn.Flatten()
        # Dynamically compute the input size for the first Linear layer
        self._to_linear = None
        self._get_flatten_size()
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def _get_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            x = self.features(x)
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


model = Mood(num_classes=len(full_dataset.classes))
torch.save(model.state_dict(), 'pet_mood_cnn.pth')


