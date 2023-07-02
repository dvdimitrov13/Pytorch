import json
import os
import random
import glob
from copy import deepcopy
import re

import ipywidgets as widgets
from IPython.display import display, clear_output
from ax.core.parameter import RangeParameter
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import render
from ax.service.utils.report_utils import get_standard_plots
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import lr_scheduler
from torchsummary import summary  # Prints out model summaries
from tqdm import tqdm

from utils.dataloader import create_data_loaders


class ModelTrainer:
    def __init__(self, init_model):
        self.device = self._get_device()
        self.init_model = init_model
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.label_mapping = None
        ## History
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def _init_model(self, parameters={}):
        # Set model
        self.model = self.init_model(parameters)
        self.model = self.model.to(self.device)

        # Reinitialize the history
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def _get_device(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
        return device

    def summary(self):
        if self.model is None:
            self._init_model()

        summary(self.model, input_size=(3, 224, 224))

    def save(self, model_path):
        """
        Save the model's state dictionary.
        Args:
            model_path (str): Path where the model will be saved.
        """
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


    def fit(self, train_loader, val_loader, parameters, patience=5, seed=420, verbose=True):
        torch.manual_seed(seed)

        if self.model is None:
            print("Model inititialized with random weights and default parameters")
            self._init_model()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=parameters.get("lr", 1e-3))
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=int(parameters.get("step_size", 10)),
            gamma=parameters.get("gamma", 0.1),
        )

        num_epochs = parameters.get("num_epochs", 30)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_weights = None

        for epoch in range(num_epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print("-" * 10)

            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                    dataloader = train_loader
                else:
                    self.model.eval()
                    dataloader = val_loader

                running_loss = 0.0
                running_corrects = 0

                iterable = tqdm(dataloader, desc=phase, unit=" batch") if verbose else dataloader

                for inputs, labels in iterable:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloader.dataset)
                
                # Calculation adjusted for architecture
                if self.device == "cuda" or self.device == "cpu":
                    epoch_acc = running_corrects.double() / len(dataloader.dataset)
                elif self.device == "mps":  # Assuming Metal (MPS) backend
                    epoch_acc = running_corrects.float() / len(dataloader.dataset)
                else:  # For TPU or any other backend
                    epoch_acc = running_corrects.to(torch.float64) / len(
                        dataloader.dataset
                    )

                if verbose:
                    print(f"\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                    print()
                    print()

                if phase == "train":
                    self.train_loss_history.append(epoch_loss)
                    self.train_acc_history.append(epoch_acc)
                else:
                    self.val_loss_history.append(epoch_loss)
                    self.val_acc_history.append(epoch_acc)

                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model_weights = deepcopy(self.model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1

            # Add the scheduler step here, after the end of the training phase and before the validation phase
            scheduler.step()

            if verbose:
                print()

            if patience_counter >= patience:
                if verbose:
                    print(
                        f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss."
                    )
                break

        if verbose:
            print("Training complete.")
            print()

        self.model.load_state_dict(best_model_weights)

        return self.model

    def history(self):
        # Move data to CPU and convert to NumPy arrays
        train_loss_history_np = np.array(self.train_loss_history)
        val_loss_history_np = np.array(self.val_loss_history)
        train_acc_history_np = [acc.cpu().numpy() for acc in self.train_acc_history]
        val_acc_history_np = [acc.cpu().numpy() for acc in self.val_acc_history]

        # Plot loss and accuracy history
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss history
        axes[0].plot(train_loss_history_np, label="Train Loss", color="blue")
        axes[0].plot(val_loss_history_np, label="Validation Loss", color="orange")
        axes[0].set_title("Loss History")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        # Plot accuracy history
        axes[1].plot(train_acc_history_np, label="Train Accuracy", color="blue")
        axes[1].plot(val_acc_history_np, label="Validation Accuracy", color="orange")
        axes[1].set_title("Accuracy History")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        plt.show()

        return {
            "train_loss": train_loss_history_np,
            "val_loss": val_loss_history_np,
            "train_acc": train_acc_history_np,
            "val_acc": val_acc_history_np,
        }

    def eval(self, val_loader, label_mapping, plot=False):
        if self.model is None:
            self._init_model()

        self.model.eval()  # Set the model to evaluation mode

        # Define the number of classes in your dataset
        num_classes = len(label_mapping)

        # Initialize the confusion matrix and accuracy variables
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        with torch.no_grad(), tqdm(
            total=len(val_loader), desc="Validation", unit=" batch"
        ) as pbar:
            for data in val_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                # Update the confusion matrix
                conf_matrix += confusion_matrix(
                    labels.cpu().numpy(),
                    predicted.cpu().numpy(),
                    labels=np.arange(num_classes),
                )

                # Update accuracy variables
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update y_true and y_pred
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                pbar.update()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")

        print("Confusion Matrix:")
        print(conf_matrix)

        # Create a list of class labels
        class_labels = [label_mapping[i] for i in range(num_classes)]
        if plot:
            # Visualize the confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                conf_matrix,
                annot=True,
                cmap="Blues",
                fmt="d",
                xticklabels=class_labels,
                yticklabels=class_labels,
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.show()

        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_labels))

        standard_error = 0.005  # Reflects the deviation between runs that comes from the non-deterministic nature of deep learning pipelines

        return {"accuracy": (accuracy, standard_error)}

    def optimize(
        self,
        parametrization,
        folder,
        csv_file,
        train_evaluate=None,
        num_trials=50,
        checkpoint_interval=5,
        optimize_stages=("transfer_learning", "fine_tuning"),
        resume=False,
        resume_path=None,
    ):
        if train_evaluate is None:

            def train_evaluate(
                parameters,
            ):
                random.seed(420)

                batch_size = parameters.get("batchsize", 32)

                train_transforms = A.Compose(
                    [
                        A.RandomResizedCrop(height=224, width=224, scale=(0.5, 1.0)),
                        A.HorizontalFlip(p=parameters.get("p_horizontal_flip", 0.5)),
                        A.Rotate(limit=30, p=parameters.get("p_rotate", 0.5)),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2,
                            contrast_limit=0.2,
                            p=parameters.get("p_random_brightness_contrast", 0.5),
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=parameters.get("p_hue_saturation_value", 0.5),
                        ),
                        A.GaussianBlur(p=parameters.get("p_gaussian_blur", 0.3)),
                        A.GaussNoise(
                            var_limit=(10.0, 50.0),
                            p=parameters.get("p_gauss_noise", 0.3),
                        ),
                        A.CoarseDropout(
                            max_holes=8,
                            max_height=16,
                            max_width=16,
                            min_holes=2,
                            min_height=8,
                            min_width=8,
                            fill_value=0,
                            p=parameters.get("p_coarse_dropout", 0.5),
                        ),
                        ToTensorV2(),
                    ]
                )

                val_transforms = A.Compose(
                    [A.Resize(height=224, width=224), ToTensorV2()]
                )

                train_loader, val_loader, label_mapping = create_data_loaders(
                    folder=folder,
                    csv_file=csv_file,
                    batch_size=batch_size,
                    train_transforms=train_transforms,
                    val_transforms=val_transforms,
                    albumentations=True,
                )

                self.train_loader = train_loader
                self.val_loader = val_loader
                self.label_mapping = label_mapping

                # Initialize net
                self._init_model(parameters)
                # self.summary()

                if "transfer_learning" in optimize_stages:
                    # Step 1: Train the classification head with a frozen backbone
                    print("Training the classification head with a frozen backbone...")
                    self.fit(train_loader, val_loader, parameters=parameters, verbose=False)

                if "fine_tuning" in optimize_stages:
                    # Step 2: Unfreeze the backbone and train with a lower learning rate
                    print(
                        "Unfreezing the backbone and training with a lower learning rate..."
                    )
                    self.model.freeze_backbone(mode="fine-tune")
                    # Reduce the learning rate for fine tuning
                    parameters["lr"] = 1e-5
                    parameters["num_epochs"] = 1
                    # fine-tune
                    self.fit(train_loader, val_loader, parameters=parameters, verbose=False)

                return self.eval(val_loader, label_mapping)

        # Create an AxClient with the search space
        ax_client = AxClient()
        ax_client.create_experiment(
            parametrization, objective_name="accuracy", minimize=False
        )

        # Create a folder for the current experiment if resume is not set to True
        if not resume:
            exp_idx = 1
            while True:
                exp_folder = f"exp{exp_idx}"
                if not os.path.exists(exp_folder):
                    os.makedirs(exp_folder)
                    break
                exp_idx += 1
        else:
            exp_folders = sorted(glob.glob("exp*"), key=lambda x: int(x[3:]) if x[3:].isdigit() else -1)
            if exp_folders:
                exp_folder = exp_folders[-1]
            else:
                raise ValueError("No experiment folders found. Please set resume to False or provide a valid resume_path.")

        checkpoint_folder_name = "checkpoints"
        checkpoint_file = "ax_client_checkpoint_{}.json"

        # Load the initial checkpoint if it exists and resume is set to True
        start_epoch = 0
        if resume:
            if resume_path is not None:
                exp_folder = resume_path

            checkpoint_folder = os.path.join(exp_folder, checkpoint_folder_name)
            print(checkpoint_folder)
            if os.path.exists(checkpoint_folder):
                def checkpoint_key(filename):
                    # Extract the checkpoint number from the filename
                    checkpoint_num = int(re.search(r"ax_client_checkpoint_(\d+).json", filename).group(1))
                    return checkpoint_num

                checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_folder, "ax_client_checkpoint_*.json")), key=checkpoint_key)

                if checkpoint_files:
                    latest_checkpoint_path = checkpoint_files[-1]
                    ax_client = AxClient.load_from_json_file(filepath=latest_checkpoint_path)

                    # Read the starting epoch from the checkpoint file
                    with open(latest_checkpoint_path, "r") as f:
                        checkpoint_data = json.load(f)
                        start_epoch = checkpoint_data["epoch"]
                else:
                    print("Warning: No checkpoint files found in the specified resume path.")
            else:
                print("Warning: Checkpoints folder not found in the specified resume path.")

        checkpoint_folder = os.path.join(exp_folder, checkpoint_folder_name)
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        # Start the optimization loop from the loaded epoch
        for i in range(start_epoch, num_trials):
            print(f"Iteration {i+1}/{num_trials}")
            try:
                print(ax_client.generation_strategy.trials_as_df.iloc[:, :-1])
            except:
                pass

            # Generate, run, and complete the next trial
            parameters, trial_index = ax_client.get_next_trial()
            accuracy = train_evaluate(parameters)
            ax_client.complete_trial(trial_index=trial_index, raw_data=accuracy)

            # Save a checkpoint every checkpoint_interval trials
            if (i + 1) % checkpoint_interval == 0:
                print("Saving trial:", i+1)
                checkpoint_temp_path = os.path.join(checkpoint_folder, "ax_client_checkpoint_temp.json")
                ax_client.save_to_json_file(filepath=checkpoint_temp_path)

                # Update the epoch information and save the final checkpoint
                with open(checkpoint_temp_path, "r") as f:
                    checkpoint_data = json.load(f)
                    checkpoint_data["epoch"] = i + 1
                checkpoint_number = len(glob.glob(os.path.join(checkpoint_folder, "ax_client_checkpoint_*.json")))
                checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file.format(checkpoint_number))
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint_data, f)

                # Remove the temporary checkpoint file
                os.remove(checkpoint_temp_path)

        # Get the best parameters
        best_parameters, values = ax_client.get_best_parameters()

        print()
        print("Training final model using best parameters:")
        print()
        # train_evaluate(best_parameters)

        print("The best model parameters are: ", best_parameters)
        acc, _ = values
        print("With a validation accurcy of: ", acc)

        render(ax_client.get_optimization_trace())

        print(ax_client.generation_strategy)

        if start_epoch < num_trials:

            # Get continuous parameters
            continuous_parameters = [
                param_name
                for param_name, param in ax_client.experiment.search_space.parameters.items()
                if isinstance(param, RangeParameter)
            ]

            def render_contour_plot(param_x, param_y):
                plot = ax_client.get_contour_plot(param_x=param_x, param_y=param_y, metric_name="accuracy")
                render(plot)

            # Create dropdown widgets for selecting continuous parameters
            param_x_widget = widgets.Dropdown(options=continuous_parameters, value=continuous_parameters[0], description="Param X:")
            param_y_widget = widgets.Dropdown(options=continuous_parameters, value=continuous_parameters[1], description="Param Y:")

            # Create a button to trigger the plot update
            update_button = widgets.Button(description="Update Plot")

            # Define the update function
            def on_update_button_click(button):
                clear_output(wait=True)
                display(param_x_widget, param_y_widget, update_button)
                render_contour_plot(param_x_widget.value, param_y_widget.value)

            # Connect the button to the update function
            update_button.on_click(on_update_button_click)

            # Display the widgets and the initial plot
            display(param_x_widget, param_y_widget, update_button)
            render_contour_plot(param_x_widget.value, param_y_widget.value)

        else:
            print("Contour plots require at least one optimization iteration!")

        return acc, best_parameters
