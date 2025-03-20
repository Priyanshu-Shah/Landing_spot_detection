import sys
import os
import moviepy.editor as mpy
import pims
import numpy as np
import torch as tr
from matplotlib.cm import hot
from matplotlib import pyplot as plt
from functools import partial
from lycon import resize, Interpolation

from main import getModel
from argparse import ArgumentParser

def getArgs():
    parser = ArgumentParser()
    parser.add_argument("task", help="regression for Depth / classification for HVN")
    parser.add_argument("in_video", help="Path to input video")
    parser.add_argument("out_video", help="Path to output video")

    # Model stuff
    parser.add_argument("--model", type=str)
    parser.add_argument("--weights_file")
    parser.add_argument("--data_dims", default="rgb")
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

def minMaxNormalizeFrame(frame):
    Min, Max = np.min(frame), np.max(frame)
    frame -= Min
    frame /= (Max - Min)
    frame *= 255
    return np.float32(frame)

def make_frame(t, model, video, fps, inputShape, args):
    t = min(int(t * fps), len(video) - 1)
    currentFrame = np.array(video[t])
    outH, outW = currentFrame.shape[0:2]
    inH, inW = inputShape

    # Resize and normalize the frame
    image = resize(currentFrame, height=inH, width=inW, interpolation=Interpolation.CUBIC)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.transpose(image, (0, 3, 1, 2))  # Change shape to [batch_size, channels, height, width]

    # Convert to float32 before passing to the model
    trImage = tr.from_numpy(image).to("cuda" if tr.cuda.is_available() else "cpu", dtype=tr.float32)
    trResult = model.forward(trImage)
    npResult = trResult.detach().cpu().numpy()[0]
    npResult = minMaxNormalizeFrame(npResult)

    # Visualize depth or classification results
    if args.label_dims == ["depth"]:
        frame = hot(npResult)[..., :3]  # Apply the hot colormap
        frame = minMaxNormalizeFrame(frame)
        frame = frame[0]
    else:
        npResult =  npResult.transpose((1, 2, 0))
        print("npResult shape:", npResult.shape)
        hvnFrame = np.argmax(npResult, axis=-1)
        frame = np.zeros((*hvnFrame.shape, 3), dtype=np.float32)
        frame[np.where(hvnFrame == 0)] = (0, 255, 0)  # Green for class 0
        frame[np.where(hvnFrame == 1)] = (255, 0, 0)  # Red for class 1
        frame[np.where(hvnFrame == 2)] = (0, 0, 255)  # Blue for class 2
    
    # Resize the frame back to the original dimensions
    print("frame shape:", frame.shape)
    return frame

def main():
    args = getArgs()
    video = pims.Video(args.in_video)
    fps = video.frame_rate
    duration = len(video) / fps 
    print(f"In video: {args.in_video}. Out video: {args.out_video}. FPS: {fps}. Duration: {duration}.")

    dIn = 3  # Assuming RGB input
    dOut = 3 if args.task == "classification" else 1  # 3 for classification, 1 for regression
    model = getModel(args, dIn, dOut)
    #model.load_state_dict(tr.load(args.weights_file))
    model.to("cuda" if tr.cuda.is_available() else "cpu")
    model.eval()

    inH, inW = 240, 320
    frameCallback = partial(make_frame, model=model, video=video, fps=fps, inputShape=(inH, inW), args=args)
    clip = mpy.VideoClip(frameCallback, duration=duration)
    ffmpeg_params = ["-crf", "0", "-preset", "veryslow", "-tune", "film"]
    clip.write_videofile(args.out_video, fps=fps, verbose=False, ffmpeg_params=ffmpeg_params)

    tr.cuda.empty_cache()

if __name__ == "__main__":
    main()