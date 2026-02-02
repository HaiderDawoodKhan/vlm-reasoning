import argparse
import os
import glob
import random
import yaml

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import OmniMod.tasks as tasks
from OmniMod.common.config import Config
from OmniMod.common.dist_utils import get_rank, init_distributed_mode
from OmniMod.common.logger import setup_logger
from OmniMod.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from OmniMod.common.registry import registry
from OmniMod.common.utils import now

# imports modules for registration
from OmniMod.models import *
from evaluate import *
from OmniMod.processors import *
from OmniMod.runners import *
from OmniMod.tasks import *


DATASETS = {
    "vqav2": {
        "image_train": {
            "image_path": "/home/csalt/Haider/DVLM/Download-Datasets/VQAv2/train2014",
            "ann_path": "/home/csalt/Haider/DVLM/datasets_image/VQAv2/annotation/train_data.json"
        },
        "image_val": {
            "image_path": "/home/csalt/Haider/DVLM/Download-Datasets/VQAv2/train2014",
            "ann_path": "/home/csalt/Haider/DVLM/datasets_image/VQAv2/annotation/train_data.json"
        },
    },
    "scienceqa" : {
        "image_train" : {
            "image_path" : "/home/csalt/Haider/DVLM/Download-Datasets/ScienceQA/train",
            "ann_path" : "/home/csalt/Haider/DVLM/datasets_image/ScienceQA/annotation/train_data.json"
        },
        "image_val" : {
            "image_path" : "/home/csalt/Haider/DVLM/Download-Datasets/ScienceQA/test",
            "ann_path" : "/home/csalt/Haider/DVLM/datasets_image/ScienceQA/annotation/test_data.json"
        },
    },
    "mmmu" : {
        "image_train" : {
            "image_path" : "/home/csalt/Haider/DVLM/datasets_image/MMMU/images",
            "ann_path" : "/home/csalt/Haider/DVLM/datasets_image/MMMU/annotation/dev_data.json"
        },
        "image_val" : {
            "image_path" : "/home/csalt/Haider/DVLM/datasets_image/MMMU/images",
            "ann_path" : "/home/csalt/Haider/DVLM/datasets_image/MMMU/annotation/validation_data.json"
        },
    },
    "mmstar" : {
        "image_train" : {
            "image_path" : "/home/csalt/Haider/DVLM/datasets_image/MMStar/images",
            "ann_path" : "/home/csalt/Haider/DVLM/datasets_image/MMStar/annotation/val_data.json"
        },
        "image_val" : {
            "image_path" : "/home/csalt/Haider/DVLM/datasets_image/MMStar/images",
            "ann_path" : "/home/csalt/Haider/DVLM/datasets_image/MMStar/annotation/val_data.json"
        },
    },
}


def list_of_str(arg):
    return list(map(str, arg.split(',')))

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to train configuration file.")
    parser.add_argument("--cfg-eval-path", required=False, help="path to evaluation configuration file.")
    parser.add_argument("--eval-dataset", type=list_of_str, default='video_val', help="dataset to evaluate")
    parser.add_argument("--swap-order", action='store_true', help="whether to swap the order of image and text input.")
    parser.add_argument("--coconut", action='store_true', help="use coconut reasoning.")
    parser.add_argument("--multinut", action='store_true', help="use multimodal coconut reasoning.")
    parser.add_argument("--modified_multinut", action='store_true', help="use modified multimodal coconut reasoning.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="directory to save the results.")
    parser.add_argument("--max-epoch", type=int, default=10, help="maximum number of training epochs.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="number of warmup steps for learning rate scheduler.")
    parser.add_argument("--iters-per-epoch", type=int, default=37, help="number of iterations per epoch.")
    parser.add_argument("--ckpt-dir", type=str, required=False, help="path to checkpoint directory to resume training from.")
    parser.add_argument("--ckpt", type=str, required=False, help="path to checkpoint to resume training from.")
    parser.add_argument("--job-name", type=str, default="OmniMod_job", help="name of the training job.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DATASETS.keys()),
        help="dataset to use (must be one of the keys in DATASETS)",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls

def get_checkpoint(ckpt_dir: str) -> str:
    ckpt_dir = os.path.expanduser(ckpt_dir)
    print(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint base directory not found: {ckpt_dir}")

    # Pick newest run subfolder inside ckpt_dir (prefer numeric names like timestamps)
    run_dirs = [
        os.path.join(ckpt_dir, d)
        for d in os.listdir(ckpt_dir)
        if os.path.isdir(os.path.join(ckpt_dir, d))
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run subdirectories found in: {ckpt_dir}")

    def run_sort_key(p: str):
        name = os.path.basename(p)
        if name.isdigit():
            return (1, int(name))
        return (0, os.path.getmtime(p))

    run_dir = sorted(run_dirs, key=run_sort_key)[-1]

    # Find checkpoints in that run directory
    ckpt_paths = glob.glob(os.path.join(run_dir, "*.pth"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No .pth checkpoints found in: {run_dir}")

    def ckpt_sort_key(p: str):
        base = os.path.basename(p)
        stem = os.path.splitext(base)[0]
        try:
            step = int(stem.split("_")[-1])  # checkpoint_123 -> 123
            return (1, step)
        except Exception:
            return (0, os.path.getmtime(p))

    return sorted(ckpt_paths, key=ckpt_sort_key)[-1]
    

def main():

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    cfg = Config(args)

    
    if args.ckpt and args.ckpt_dir:
        raise ValueError("Cannot use both ckpt and ckpt_dir at the same time.")
    
    if args.ckpt:
        cfg.model_cfg.ckpt = args.ckpt
    
    if args.ckpt_dir:
        cfg.model_cfg.ckpt = get_checkpoint(args.ckpt_dir)
        
    if args.coconut:
        cfg.model_cfg.use_coconut = True
        
    if args.multinut and args.modified_multinut:
        raise ValueError("Cannot use both multinut and modified_multinut at the same time.")
    
    if args.multinut:
        cfg.model_cfg.use_multimodal_coconut = True
    if args.modified_multinut:
        cfg.model_cfg.use_modified_multinut_with_attention = True
        
    if args.dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{args.dataset}'. Available: {', '.join(sorted(DATASETS.keys()))}"
        )
    
    if args.swap_order:
        cfg.datasets_cfg.image_train.build_info.swap_order = True
        
    if "image_train" in cfg.datasets_cfg and "build_info" in cfg.datasets_cfg.image_train:
        cfg.datasets_cfg.image_train.build_info.dataset_name = args.dataset
        cfg.datasets_cfg.image_train.build_info.image_path = DATASETS[args.dataset]["image_train"]["image_path"]
        cfg.datasets_cfg.image_train.build_info.ann_path = DATASETS[args.dataset]["image_train"]["ann_path"]
    else:
        raise KeyError("Config is missing datasets.image_train.build_info")
    
    cfg.run_cfg.output_dir = args.output_dir
    cfg.run_cfg.job_name = args.job_name
    cfg.run_cfg.warmup_steps = args.warmup_steps
    cfg.run_cfg.iters_per_epoch = args.iters_per_epoch
    cfg.run_cfg.max_epoch = args.max_epoch
    
    print(cfg)

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    # # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()

    if hasattr(args, 'cfg_eval_path'):
        args.cfg_path = args.cfg_eval_path

        model_path = "OmniMod/{}/{}".format(cfg.run_cfg.output_dir, job_id)
        ckpt_paths = glob.glob(os.path.join(model_path, "*.pth"))
        ckpt_names = [os.path.basename(ckp_path) for ckp_path in ckpt_paths]
        last_ckpt_name = sorted(ckpt_names, key=lambda x: int(x.split(".")[0].split("_")[-1]))[-1]
        last_ckpt_path = os.path.join(model_path, last_ckpt_name)

        with open(args.cfg_path) as f:
            eval_cfg = yaml.load(f, Loader=yaml.FullLoader)
            eval_cfg["model"]["ckpt"] = last_ckpt_path

        with open(args.cfg_path, "w") as f:
            yaml.dump(
                eval_cfg, stream=f, default_flow_style=False, sort_keys=False
            )
        
        print("Evaluating...........")
        evaluate(args)
        print("Done!")

if __name__ == "__main__":
    main()