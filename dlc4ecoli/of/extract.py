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
    parser = ArgumentParser(
        "Arguments for optical flow stats extraction",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_folder", type=str, help="Input video folder")

    parser.add_argument("model", type=str, help="Optical flow model")

    parser.add_argument("--output_folder", type=str, help="Output folder", default="of")

    parser.add_argument(
        "--ext",
        default="mp4",
        help="Video extension",
    )

    parser.add_argument("--batch-size", default=8, type=int, help="Batch size")

    parser.add_argument(
        "--device",
        default="cuda:0",
        choices=("cuda:0", "cuda:1", "cpu"),
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


def run_optical_flow_single(model_name, video, model, batch_size, device, scale=1.0):
    try:
        video_info = get_video_info(video)
    except OSError as e:
        warnings.warn("Error reading video: " + str(e), UserWarning)
        return

    n_frames = video_info.total_frames

    height = video_info.height
    width = video_info.width

    print(f"Original size: ({height}, {width})")

    if model_name.startswith("raft"):
        # height/width need to be divisible by 8
        size = (int((height * scale) // 8 * 8), int((width * scale) // 8 * 8))
    else:
        size = (int(height * scale), int(width * scale))

    print(f"\t {n_frames} frames. Inference size: {size}")

    n_batches = (n_frames - 1) // batch_size

    # mean, SD, skew, kurtosis, median
    flow_stats = torch.zeros((n_frames, 4)).to(device)

    for i in tqdm(range(n_batches)):
        with torch.no_grad():
            # raw_batch: N x C x W x H
            raw_batch = load_n_frames(
                video,
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
    args = parse_args()

    print("Configuration")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    videos = sorted(glob(f"{args.input_folder}/*.mp4"))

    output_folder = f"{args.input_folder}/{args.output_folder}"
    os.makedirs(output_folder, exist_ok=True)

    print(f"Found {len(videos)} videos.")
    assert len(videos) > 0

    model = nn.DataParallel(
        load_model(args.model, url=args.sea_raft_url, config=args.sea_raft_cfg).to(
            args.device
        ),
        device_ids=[0, 1],
    )
    model.to(args.device)
    model.eval()

    summary(model, depth=1)

    for i, video in enumerate(videos):
        camera = os.path.basename(Path(video).parent)

        of_camera_folder = f"{output_folder}/{camera}"
        os.makedirs(of_camera_folder, exist_ok=True)

        stem = Path(video).stem

        output_file = f"{of_camera_folder}/{stem}.pt"

        print(f"Processing video {i+1}/{len(videos)} ({camera}/{stem})...")

        if os.path.isfile(output_file) and not args.overwrite:
            warnings.warn(f"{output_file} already exists.", UserWarning)
            continue

        flow_stats = run_optical_flow_single(
            video=video,
            model=model,
            batch_size=args.batch_size,
            device=args.device,
            model_name=args.model,
            scale=args.scale,
        )

        torch.save(flow_stats, output_file)

    with open(f"{output_folder}/cfg.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    of_agg = build_summary(output_folder, agg=args.agg)

    for i, stat in enumerate(STATS):
        of_agg.rename(columns={f"{args.agg}_{stat}": f"of_{stat}"}).loc[
            :, ["video", f"of_{stat}", "camera", "Day"]
        ].to_csv(f"data/of/summary_of_{stat}.csv")


if __name__ == "__main__":
    main()
