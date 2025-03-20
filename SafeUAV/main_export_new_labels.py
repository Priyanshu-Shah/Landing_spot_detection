import sys
import os
import h5py
import numpy as np
import torch
from main import getModel
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from loss import l2_loss, classification_loss

def getArgs():
    parser = ArgumentParser()
    parser.add_argument("task", help="regression for Depth / classification for HVN")
    parser.add_argument("dataset_path", help="Path to dataset")

    parser.add_argument("--batch_size", default=10, type=int)

    # Model stuff
    parser.add_argument("--model", type=str)
    parser.add_argument("--weights_file")
    parser.add_argument("--label_dims")

    args = parser.parse_args()
    assert args.weights_file is not None
    args.weights_file = os.path.abspath(args.weights_file)
    assert args.task in ("classification", "regression")
    assert args.model in ("unet_big_concatenate", "unet_tiny_sum")

    args.data_dims = ["rgb"]
    if args.task == "classification":
        args.label_dims = ["hvn_gt_p1"]
    else:
        args.label_dims = ["depth"]
    return args

class CustomDataset(Dataset):
    def __init__(self, dataset_path, data_dims, label_dims, transform=None):
        # Load and preprocess your data here
        self.data = ...  # Load data from dataset_path
        self.labels = ...  # Load labels from dataset_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

class ExportCallback:
    def __init__(self, reader, originalLabelsName, newLabelsName):
        self.file = h5py.File(f"{newLabelsName}.h5", "w")
        dType = reader["train"][originalLabelsName].dtype
        trainShape = list(reader["train"][originalLabelsName].shape)
        valShape = list(reader["validation"][originalLabelsName].shape)
        trainShape[1:3] = 240, 320
        valShape[1:3] = 240, 320

        self.newLabelsName = newLabelsName

        self.file.create_group("train")
        self.file.create_group("validation")
        self.file["train"].create_dataset(newLabelsName, dtype=dType, shape=trainShape)
        self.file["validation"].create_dataset(newLabelsName, dtype=dType, shape=valShape)

        self.group = None
        self.index = None

    def setGroup(self, group):
        assert group in ("train", "validation")
        self.group = group
        self.index = 0

    def onIterationEnd(self, results):
        dataset = self.file[self.group][self.newLabelsName]
        for i, result in enumerate(results):
            if "hvn" in self.newLabelsName:
                result = np.argmax(result, axis=-1)
            else:
                result = result[..., 0]
            dataset[self.index] = result
            self.index += 1
            if self.index % 10 == 0:
                print(f"Processed {self.index} samples.")

def main():
    args = getArgs()

    hvnTransform = "hvn_two_dims" if args.task == "regression" else "identity_long"

    trainDataset = CustomDataset(args.dataset_path, data_dims=args.data_dims, label_dims=args.label_dims, transform=None)
    valDataset = CustomDataset(args.dataset_path, data_dims=args.data_dims, label_dims=args.label_dims, transform=None)

    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=False)
    valLoader = DataLoader(valDataset, batch_size=args.batch_size, shuffle=False)

    dIn = len(args.data_dims)
    dOut = 3 if args.task == "classification" else len(args.label_dims)
    model = getModel(args, dIn=dIn, dOut=dOut)
    model.load_state_dict(torch.load(args.weights_file))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    criterion = l2_loss if args.task == "regression" else classification_loss

    modelName = "tiny" if "tiny" in args.model else "big"
    newLabelDim = f"depth_{modelName}_it1" if args.label_dims[0] == "depth" else f"hvn_{modelName}_it1_p1"
    callback = ExportCallback(trainDataset, args.label_dims[0], newLabelDim)

    # Process training data
    callback.setGroup("train")
    for batch in trainLoader:
        data, labels = batch
        data = data.to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            results = model(data)
        callback.onIterationEnd(results.cpu().numpy())

    # Process validation data
    callback.setGroup("validation")
    for batch in valLoader:
        data, labels = batch
        data = data.to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            results = model(data)
        callback.onIterationEnd(results.cpu().numpy())

if __name__ == "__main__":
    main()