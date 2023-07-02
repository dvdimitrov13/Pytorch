import os

# Set the GPU device index you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the second GPU

import torch.nn as nn
from torchvision.models.resnet import ResNet50_Weights, resnet50

from utils.models import CustomModel
from utils.trainers import ModelTrainer


def init_model(parameters):
    # Define the backbone and head
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = backbone.fc.in_features
    backbone = nn.Sequential(*list(backbone.children())[:-1])

    Hs = parameters.get("Hs", 1024)
    dropout = parameters.get("dropout", 0.2)

    head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, Hs),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(Hs, int(Hs / 2)),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(int(Hs / 2), 3),
    )

    model = CustomModel(backbone, head)

    model.freeze_backbone()

    return model


if __name__ == '__main__':
    folder = "./data/background_data/background_data"
    csv_file = "./data/background_data/background_data.csv"

    parametrization = [
        {"name": "lr", "type": "range", "bounds": [1e-4, 1e-1], "log_scale": True},
        # {"name": "batchsize", "type": "choice", "values": [32, 64, 96]},
        # {"name": "stepsize", "type": "choice", "values": [5, 10]},
        # {"name": "Hs", "type": "choice", "values": [256, 512, 1024]},
        {"name": "dropout", "type": "range", "bounds": [0.2, 0.4]},
        # {"name": "p_horizontal_flip", "type": "range", "bounds": [0.2, 0.8]},
        # {"name": "p_rotate", "type": "range", "bounds": [0.2, 0.8]},
        # {"name": "p_random_brightness_contrast", "type": "range", "bounds": [0.2, 0.8]},
        # {"name": "p_hue_saturation_value", "type": "range", "bounds": [0.2, 0.8]},
        # {"name": "p_gaussian_blur", "type": "range", "bounds": [0.1, 0.5]},
        # {"name": "p_gauss_noise", "type": "range", "bounds": [0.1, 0.5]},
        # {"name": "p_coarse_dropout", "type": "choice", "values": [0.0, 0.5]},
        {"name": "num_epochs", "type": "choice", "values": [1]},
    ]

    trainer = ModelTrainer(init_model)
    # This returns accuracy and best model params but also sets the model to the best model
    acc, best_params = trainer.optimize(parametrization=parametrization, folder=folder, csv_file=csv_file, num_trials = 10)
    trainer.history()
    trainer.eval(
        val_loader=trainer.val_loader, label_mapping=trainer.label_mapping, plot=True
    )
    trainer.save('models/background.pt')

    ## TODO: Add a folder structure to the optmizer as well as a resume variable
