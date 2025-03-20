import numpy as np
from plotter import plotter, show_plots

def testDataset(reader, args):
    generator = reader.iterate("train", miniBatchSize=args.batch_size)
    steps = reader.getNumIterations("train", miniBatchSize=args.batch_size)
    from functools import partial

    for items in generator:
        datas, labels = items
        MB = datas.shape[0]

        for i in range(MB):
            data = datas[i]
            label = labels[i]
            totalItems = len(args.data_dims) + len(args.label_dims)
            plotter(data, args.data_dims, reader.hvnTransform, startingIndex=1, totalItems=totalItems)
            ix = len(args.data_dims) + 1
            plotter(label, args.label_dims, reader.hvnTransform, startingIndex=ix, totalItems=totalItems)
            show_plots()