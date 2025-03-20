import numpy as np
import matplotlib.pyplot as plt

def hvnPlotter(x, hvnTransform):
    hvn = np.zeros((*x.shape[0:2], 3), dtype=np.uint8)
    if hvnTransform in ("identity_long", "identity"):
        # H, V, N
        hvn[np.where(x == 0)] = (0, 255, 0)
        hvn[np.where(x == 1)] = (255, 0, 0)
        hvn[np.where(x == 2)] = (0, 0, 255)
    elif hvnTransform == "hvn_two_dims":
        # N, H, V
        hvn[..., 0:3] = (0, 0, 255)
        hvn[np.where(x[..., 0] == 1)] = (0, 255, 0)
        hvn[np.where(x[..., 1] == 1)] = (255, 0, 0)
    return hvn

def plot_image(image, cmap=None, new_figure=True, show_axis=True, title=None, axis=None):
    if new_figure:
        plt.figure()
    plt.imshow(image, cmap=cmap)
    if not show_axis:
        plt.axis('off')
    if title:
        plt.title(title)
    if axis:
        plt.subplot(*axis)

def show_plots():
    plt.show()

def figure_set_size(size):
    plt.gcf().set_size_inches(*size)

def save_figure(filename):
    plt.savefig(filename)

def plotter(data, dims, hvnTransform, startingIndex, totalItems):
    j = 0
    for i, dim in enumerate(dims):
        if dim == "rgb":
            rgb = data[..., j:j + 3]
            plot_image(rgb, new_figure=(startingIndex == 1 and i == 0), show_axis=False,
                       title="RGB", axis=(1, totalItems, startingIndex + i))
            j += 3
        elif dim == "hvn_gt_p1":
            if hvnTransform in ("identity", "identity_long"):
                hvn = hvnPlotter(data[..., j], hvnTransform)
            elif hvnTransform == "hvn_two_dims":
                hvn = hvnPlotter(data[..., j:j + 2], hvnTransform)
            plot_image(hvn, new_figure=(startingIndex == 1 and i == 0), show_axis=False, title="HVN",
                       axis=(1, totalItems, startingIndex + i))
            j += 1
        elif dim == "depth":
            depth = data[..., j]
            plot_image(depth, cmap="hot", new_figure=(startingIndex == 1 and i == 0), show_axis=False,
                       title="Depth", axis=(1, totalItems, startingIndex + i))
            j += 1

class PlotCallback:
    def __init__(self, args):
        self.args = args
        self.hvnTransform = "hvn_two_dims" if args.task == "regression" else "identity_long"

    def doPlot(self, data, label, result):
        totalItems = len(self.args.data_dims) + len(self.args.label_dims) + 2
        plotter(data, self.args.data_dims, self.hvnTransform, startingIndex=1, totalItems=totalItems)
        ix = len(self.args.data_dims) + 1
        plotter(label, self.args.label_dims, self.hvnTransform, startingIndex=ix, totalItems=totalItems)

        if self.args.task == "regression":
            diffImage = np.abs(label - result)[..., 0]
            resultImage = result[..., 0]
            cmap = "hot"
        elif self.args.task == "classification":
            result = np.argmax(result, axis=-1)
            resultImage = hvnPlotter(result, hvnTransform="identity")
            diffImage = np.uint8(label[..., 0] == result)
            cmap = "gray"
        else:
            raise ValueError("Unsupported task type")

        plot_image(diffImage, cmap=cmap, new_figure=False, show_axis=False,
                   title="Diff Image", axis=(1, totalItems, totalItems - 1))
        plot_image(resultImage, cmap=cmap, new_figure=False, show_axis=False,
                   title="Result", axis=(1, totalItems, totalItems))

    def onIterationEnd(self, **kwargs):
        if not self.args.test_plot_results and not self.args.test_save_results:
            return

        data = kwargs["data"]
        labels = kwargs["labels"]
        results = kwargs["results"]
        metrics = kwargs["metrics"]

        print(metrics)
        for i in range(len(results)):
            self.doPlot(data[i], labels[i], results[i])
            figure_set_size((20, 8))
            if self.args.test_plot_results:
                show_plots()
            if self.args.test_save_results:
                save_figure(f"result_{i}.png")