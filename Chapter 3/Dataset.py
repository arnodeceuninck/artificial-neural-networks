from torchvision import transforms, datasets
import torch
from settings import batch_size,input_shape
import os

def load_date():
    # Creating a transformers including all pre-processing components such as normalization.
    transform = transforms.Compose([transforms.Resize((input_shape[0],input_shape[1])),
                                    transforms.ToTensor()])
    trainset = datasets.ImageFolder(os.path.join('./data/15SceneData/', 'train'),transform)

    print('number of train examples ',len(trainset))
    validationset = datasets.ImageFolder(os.path.join('./data/15SceneData/', 'test'),transform)
    print('number of evaluation examples ',len(validationset))
    # Creating a loader object to read and load a batch of data
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, validation_loader
