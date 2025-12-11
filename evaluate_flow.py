import argparse
import os
import time
from typing import Dict

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torcheval.metrics import MeanSquaredError, PeakSignalNoiseRatio, Throughput

from config import config
from utils.dataset import DefaultDataset
from utils.sampler import SamplingPipeline
from utils.ssim import StructuralSimilarity


def evaluate_flow(pipeline, dataloader, num_inference_steps: int) -> Dict[str, float]:
    """Run flow-matching pipeline over the dataset and return aggregated metrics."""
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarity()
    mse = MeanSquaredError()
    throughput_metric = Throughput()

    for batch in dataloader:
        start = time.time()
        outputs = pipeline(
            num_inference_steps=num_inference_steps,
            batch_size=batch["image"].shape[0],
            output_type="np.array",
            images=batch,
        ).images
        end = time.time()

        outputs = torch.from_numpy(outputs).permute(0, 3, 1, 2)
        inputs = pipeline.preprocess(batch)

        throughput_metric.update(num_processed=batch["image"].shape[0], elapsed_time_sec=end - start)

        for inp, out in zip(inputs, outputs):
            psnr.update(inp, out)
            ssim.update(inp, out)
            mse.update(inp.flatten(), out.flatten())

    metrics = {
        "psnr": psnr.compute().item(),
        "ssim": ssim.compute().item(),
        "rmse": float(np.sqrt(mse.compute())),
        "throughput_imgs_per_sec": throughput_metric.compute().item(),
    }
    return metrics


def build_dataloader():
    dataset = DefaultDataset("./DefaultDataset", img_size=config.image_size, s_cnt=config.slices, train=False)
    loader_args = dict(batch_size=config.eval_batch_size, num_workers=4, pin_memory=True)
    return torch.utils.data.DataLoader(dataset, shuffle=False, **loader_args)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate flow-matching model on the test set.")
    parser.add_argument("--model-path", required=True, help="Path to the saved flow-matching pipeline.")
    parser.add_argument("--device", default="cuda", help="Device to run evaluation on (e.g., cuda or cpu).")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=config.num_inference_steps,
        help="Number of inference steps for the sampler.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

    pipeline = SamplingPipeline.from_pretrained(
        args.model_path, use_safetensors=True, conditioning=config.conditioning
    ).to(device)

    # Ensure the flow scheduler is loaded.
    scheduler_path = os.path.join(args.model_path, "scheduler")
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)

    dataloader = build_dataloader()
    metrics = evaluate_flow(pipeline, dataloader, num_inference_steps=args.num_steps)

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
