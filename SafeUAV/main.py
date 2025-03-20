import sys
import os
import h5py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial
from argparse import ArgumentParser
from collections import OrderedDict
from segmentation_models_pytorch import Unet, DeepLabV3Plus
from torchvision.transforms.functional import rgb_to_grayscale
from loss import l2_loss, classification_loss, mIoUMetric, metterMetric, precisionMetric, recallMetric, accuracyMetric
from unet_tiny_sum import ModelUNetTinySum

# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getArgs():
    parser = ArgumentParser()
    parser.add_argument("type", help="Type of operation (e.g., train, test, retrain, etc.)")
    parser.add_argument("task", help="Task type: regression for Depth / classification for HVN")
    parser.add_argument("dataset_path", help="Path to the dataset file (e.g., HDF5 file)")

    # Add missing arguments
    parser.add_argument("--model", required=True, help="Model type (e.g., unet_tiny_sum, unet_classic, etc.)")
    parser.add_argument("--dir", required=True, help="Directory to save results or logs")
    parser.add_argument("--data_dims", required=True, help="Data dimensions (e.g., rgb, depth, etc.)")  # Added this line
    parser.add_argument("--label_dims", required=True, help="Label dimensions (e.g., depth, hvn_gt_p1)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--optimizer", default="Adam", help="Optimizer to use (e.g., Adam, SGD, etc.)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--patience", type=int, default=4, help="Patience for learning rate scheduler")
    parser.add_argument("--factor", type=float, default=0.1, help="Factor for learning rate reduction")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--test_save_results", default=0, type=int, help="Flag to save test results")
    parser.add_argument("--test_plot_results", default=0, type=int, help="Flag to plot test results")

    args = parser.parse_args()
    assert args.type in ("test_dataset", "train", "retrain", "train_pretrained", "test")
    assert args.task in ("classification", "regression")
    if not args.type in ("test_dataset",):
        assert args.model in ("unet_big_concatenate", "unet_tiny_sum", "unet_classic", "deeplabv3plus")
    if args.type in ("retrain", "test", "train_pretrained"):
        args.weights_file = os.path.abspath(args.weights_file)
    args.data_dims = args.data_dims.split(",")  # This will now work
    args.label_dims = args.label_dims.split(",")
    args.test_save_results = bool(args.test_save_results)
    args.test_plot_results = bool(args.test_plot_results)
    if not args.patience:
        args.patience = args.num_epochs
    return args

class CustomDataset(Dataset):
    def __init__(self, dataset_path, data_dims, label_dims, transform=None):
        self.dataset_path = dataset_path
        self.data_dims = data_dims
        self.label_dims = label_dims
        self.transform = transform
        self.file = h5py.File(dataset_path, "r")
        self.data = self.file["train"][data_dims[0]]
        self.labels = self.file["train"][label_dims[0]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        # Ensure the sample has 3 dimensions (channels, height, width)
        if len(sample.shape) == 2:  # If shape is (height, width), add a channel dimension
            sample = np.expand_dims(sample, axis=0)  # Add channel dimension
        elif sample.shape[-1] == 3:  # If shape is (height, width, channels), transpose to (channels, height, width)
            sample = np.transpose(sample, (2, 0, 1))

        # Convert to float and normalize
        sample = torch.tensor(sample, dtype=torch.float32) / 255.0
        label = torch.tensor(label, dtype=torch.float32)

        # Apply any transformations
        if self.transform:
            sample = self.transform(sample)
            label = self.transform(label)

        return sample, label
    
def getResizer(args):
    if args.model == "unet_classic":
        resizer = {"rgb": (572, 572, 3), "depth": (388, 388, 1), "hvn_gt_p1": (388, 388, 1)}
    elif args.model == "deeplabv3plus":
        resizer = {"rgb": (512, 512, 3), "depth": (512, 512, 1), "hvn_gt_p1": (512, 512, 1)}
    else:
        resizer = (240, 320)
    return resizer

def getModel(args, dIn, dOut):
    if args.model == "unet_big_concatenate":
        model = Unet(encoder_name="resnet34", in_channels=dIn, classes=dOut)
    elif args.model == "unet_tiny_sum":
        model = ModelUNetTinySum(dIn=dIn, dOut=dOut, numFilters=16)
    elif args.model == "unet_classic":
        model = Unet(encoder_name="resnet34", in_channels=dIn, classes=dOut)
    elif args.model == "deeplabv3plus":
        model = DeepLabV3Plus(encoder_name="resnet34", in_channels=dIn, classes=dOut)
    model = model.to(device)
    return model

def setOptimizer(args, model):
    if not args.type in ("train", "retrain", "train_pretrained"):
        return

    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=False)
    elif args.optimizer == "Nesterov":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=True)
    elif args.optimizer == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    return optimizer

def getMetrics(args):
    if args.task == "regression":
        metrics = OrderedDict({
            "MSE": lambda x, y: np.mean((x - y) ** 2),
            "RMSE": lambda x, y: np.sqrt(np.mean((x - y) ** 2)),
            "L1 Loss": lambda x, y: np.mean(np.sum(np.abs(x - y), axis=(1, 2))),
            "Metters": metterMetric
        })
    else:
        metrics = OrderedDict({
            "mIoU": mIoUMetric,
            "Accuracy": accuracyMetric,
            "Precision": precisionMetric,
            "Recall": recallMetric
        })
    return metrics

def main():
    args = getArgs()

    resizer = getResizer(args)
    dataset = CustomDataset(args.dataset_path, data_dims=args.data_dims, label_dims=args.label_dims)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    dIn = len(args.data_dims)
    dOut = 3 if args.task == "classification" else len(args.label_dims)
    model = getModel(args, dIn=dIn, dOut=dOut)

    optimizer = setOptimizer(args, model)
    criterion = l2_loss if args.task == "regression" else classification_loss
    metrics = getMetrics(args)

    # Example training loop
    for epoch in range(args.num_epochs):
        model.train()
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{args.num_epochs} completed.")

if __name__ == "__main__":
    main()