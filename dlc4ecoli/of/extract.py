"""Extract statistics from optical flow outputs"""

import json
import os
import warnings

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import glob
from pathlib import Path

import torch

from torchinfo import summary
from torch import nn
from tqdm import tqdm

from .data import build_summary, load_n_frames, transform
from .features import get_stats, STATS
from .models import load_model
from ..utils.video import get_video_info


def parse_args():
    """Parse optical flow extraction arguments."""

    parser = ArgumentParser(
        "Arguments for optical flow stats extraction",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_dir", type=str, help="Input video folder")

    parser.add_argument(
        "--model_name", type=str, default="sea_raft", help="Optical flow model name"
    )

    parser.add_argument("--output_dir", type=str, help="Output folder", default="of")

    parser.add_argument(
        "--ext",
        default="mp4",
        help="Video file extension",
    )

    parser.add_argument("--batch-size", default=8, type=int, help="Batch size")

    parser.add_argument(
        "--device",
        default="0",
        help="torch device",
    )

    parser.add_argument(
        "--sea-raft-cfg", default="spring-M", type=str, help="RAFT eval config file"
    )

    parser.add_argument(
        "--sea-raft-url",
        default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
        type=str,
        help="HuggingFace URL to SEA-RAFT model",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite previous results"
    )

    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scaling size of the image"
    )

    parser.add_argument("--agg", choices=("mean", "median"), help="Aggregation mode")

    return parser.parse_args()


def run_optical_flow_single(
    model_name, video_path, model, batch_size, device, scale=1.0
):
    """
    Run optical flow on a video

    Parameters
    ----------
    model_name : str
        Model name
    video_path : str
        Path to a video
    model : nn.Module
        torch model
    batch_size : int
        Batch size
    device : str
        torch device
    scale : float, optional
        Image scale, by default 1.0

    Returns
    -------
    flow_stats : torch.Tensor
        mean, SD, skew, kurtosis of the optical flow for each frame
    """
    try:
        video_info = get_video_info(video_path)
    except OSError as e:
        warnings.warn("Error reading video: " + str(e), UserWarning)
        return

    n_frames = video_info.total_frames

    height = video_info.height
    width = video_info.width

    print(f"Original size: ({height}, {width})")

    if model_name.startswith("raft"):
        # height/width need to be divisible by 8 for RAFT models
        size = (int((height * scale) // 8 * 8), int((width * scale) // 8 * 8))
    else:
        size = (int(height * scale), int(width * scale))

    print(f"\t {n_frames} frames. Inference size: {size}")

    n_batches = (n_frames - 1) // batch_size

    # mean, SD, skew, kurtosis
    flow_stats = torch.zeros((n_frames, 4)).to(device)

    for i in tqdm(range(n_batches)):
        with torch.no_grad():
            # raw_batch: N x C x W x H
            raw_batch = load_n_frames(
                video_path,
                start=(i * batch_size),
                end=((i + 1) * batch_size) + 1,
            )

            batch = transform(raw_batch, size=size).to(device)

            batch1 = batch[:batch_size]
            batch2 = batch[1:]

            # batch_flows: N x 2 x W x H
            if model_name == "sea_raft":
                batch_flows = model(batch1, batch2, test_mode=True)["flow"][-1]
            else:
                batch_flows = model(batch1, batch2)[-1]

            batch_flow_stats = get_stats(batch_flows)

            # Stats
            flow_stats[i * batch_size : (i + 1) * batch_size, :] = batch_flow_stats

    return flow_stats


def main():
    """Main script"""
    args = parse_args()

    print("Configuration")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    videos = sorted(glob(f"{args.input_dir}/*.mp4"))

    output_dir = f"{args.input_dir}/{args.output_dir}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(videos)} videos.")
    assert len(videos) > 0

    model = load_model(
        model_name=args.model_name, url=args.sea_raft_url, config=args.sea_raft_cfg
    )

    device_ids = None

    if args.device == "cpu":
        device = "cpu"
    else:
        devices = args.device.split(",")

        device = f"cuda:{devices[0]}"

        device_ids = [int(d) for d in devices]

    model.to(device)
    model.eval()

    if device_ids is not None:
        model = nn.DataParallel(model, device_ids=device_ids)

    summary(model, depth=1)

    for i, video in enumerate(videos):
        camera = os.path.basename(Path(video).parent)

        of_camera_dir = f"{output_dir}/{camera}"
        os.makedirs(of_camera_dir, exist_ok=True)

        stem = Path(video).stem

        output_file = f"{of_camera_dir}/{stem}.pt"

        print(f"Processing video {i+1}/{len(videos)} ({camera}/{stem})...")

        if os.path.isfile(output_file) and not args.overwrite:
            warnings.warn(f"{output_file} already exists.", UserWarning)
            continue

        flow_stats = run_optical_flow_single(
            video_path=video,
            model=model,
            batch_size=args.batch_size,
            device=args.device,
            model_name=args.model_name,
            scale=args.scale,
        )

        torch.save(flow_stats, output_file)

    with open(f"{output_dir}/cfg.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    of_agg = build_summary(output_dir, agg=args.agg)

    for i, stat in enumerate(STATS):
        of_agg.rename(columns={f"{args.agg}_{stat}": f"of_{stat}"}).loc[
            :, ["video", f"of_{stat}", "camera", "Day"]
        ].to_csv(f"data/{args.output_dir}/summary_of_{stat}.csv")


if __name__ == "__main__":
    main()
