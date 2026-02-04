import os
import glob
import argparse
import json
import logging
from OmniMod.datasets.datasets.videounderstanding import VideoDataset
from OmniMod.datasets.datasets.imageunderstanding import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from OmniMod.common.registry import registry
from OmniMod.common.config import Config
from OmniMod.conversation.conversation import Conversation, SeparatorStyle
import torch

from OmniMod.metrics.metrics import (
    ensure_nltk_data,
    calculate_accuracy,
    calculate_bleu,
    calculate_rouge,
)

torch.cuda.empty_cache()

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)


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
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--cfg-path", required=True, help="path to evaluate configuration file.")
    parser.add_argument("--eval-dataset", type=list_of_str, default='video_val', help="dataset to evaluate")
    parser.add_argument("--swap-order", action='store_true', help="whether to swap the order of image and text input.")
    parser.add_argument("--coconut", action='store_true', help="use coconut reasoning.")
    parser.add_argument("--multinut", action='store_true', help="use multimodal coconut reasoning.")
    parser.add_argument("--modified-multinut", action='store_true', help="use modified multimodal coconut reasoning.")
    parser.add_argument("--pretrained-ckpt", action='store_true', help="use multimodal coconut reasoning.")
    parser.add_argument("--ckpt-dir", type=str, required=False, help="path to checkpoint directory to resume training from.")
    parser.add_argument("--ckpt", type=str, required=False, help="path to checkpoint to resume training from.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="directory to save the results.")
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
        "in xxx=yyy format will be merged into config file.",
    )
    return parser.parse_args()

def prepare_texts(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(conv.roles[0], '{}'.format(text)) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts

def init_model(cfg):
    logging.info('Initializing Model')
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')
    key = list(cfg.datasets_cfg.keys())[0]
    vis_processor_cfg = cfg.datasets_cfg.get(key).vis_processor.train
    text_processor_cfg = cfg.datasets_cfg.get(key).text_processor.train
    audio_processor_cfg = cfg.datasets_cfg.get(key).audio_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)
    audio_processor = registry.get_processor_class(audio_processor_cfg.name).from_config(audio_processor_cfg)
    logging.info('Initialization Finished')
    return model, vis_processor, text_processor, audio_processor, vis_processor_cfg.name

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
    

def evaluate(args):
    # Mark this process as an evaluation entrypoint.
    # Language-model wrappers use this to ensure logits are not force-cast to fp32.
    os.environ["OMNIMOD_RUN_MODE"] = "eval"
    cfg = Config(args)
    
    if args.ckpt and args.ckpt_dir:
        raise ValueError("Cannot use both ckpt and ckpt_dir at the same time.")
    
    if args.ckpt:
        cfg.model_cfg.ckpt = args.ckpt
        
    if args.ckpt_dir:
        cfg.model_cfg.ckpt = get_checkpoint(args.ckpt_dir)
        
    reasoning_string = "no_reasoning"
    if args.coconut:
        cfg.model_cfg.use_coconut = True
        reasoning_string = "coconut"
    else:
        cfg.model_cfg.use_coconut = False
    
    if args.multinut and args.modified_multinut:
        raise ValueError("Cannot use both multinut and modified_multinut at the same time.")
    
    if args.multinut:
        cfg.model_cfg.use_multimodal_coconut = True
        reasoning_string = "multinut"
    else:
        cfg.model_cfg.use_multimodal_coconut = False
        
    if args.modified_multinut:
        cfg.model_cfg.use_modified_multinut_with_attention = True
        reasoning_string = "modified_multinut"
    else:
        cfg.model_cfg.use_modified_multinut_with_attention = False
    
    model, vis_processor, text_processor, audio_processor, vis_processor_name = init_model(cfg)
    model.eval()
    conv_temp = CONV_VISION.copy()

    # Metrics rely on NLTK tokenizers; ensure they're available once.
    ensure_nltk_data()
    
    swap_order = False
    swap_order_string = "org_ord"
    if args.swap_order:
        swap_order = True
        swap_order_string = "swpd_ord"
        
    pretrained_string = ""
    if args.pretrained_ckpt:
        pretrained_string = "_pretrained"
        
    if args.dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{args.dataset}'. Available: {', '.join(sorted(DATASETS.keys()))}"
        )

    for dataset in args.eval_dataset:
        # Prefer dataset selection from DATASETS dict for file locations.
        # Keep other eval hyperparams (batch size, decoding params, etc.) from the YAML config.
        dataset_name = args.dataset

        if dataset not in DATASETS[args.dataset]:
            raise ValueError(
                f"Unknown eval split '{dataset}' for dataset '{args.dataset}'. "
                f"Available: {', '.join(sorted(DATASETS[args.dataset].keys()))}"
            )

        eval_file_path = DATASETS[args.dataset][dataset]["ann_path"]
        img_path = DATASETS[args.dataset][dataset]["image_path"]
        prompt_test = cfg.evaluation_datasets_cfg[dataset]["prompt_test"]
        batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
        max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]
        temperature = cfg.evaluation_datasets_cfg[dataset]["temperature"]
        top_p = cfg.evaluation_datasets_cfg[dataset]["top_p"]
        do_sample = cfg.evaluation_datasets_cfg[dataset]["do_sample"]
        # Audio path: take from DATASETS if provided, otherwise fall back to YAML.
        audio_path = DATASETS[args.dataset][dataset].get(
            "audio_path", cfg.evaluation_datasets_cfg[dataset].get("audio_path", None)
        )

        if vis_processor_name == 'blip2_image_eval' or vis_processor_name == 'blip2_image_train':
            data = ImageDataset(
                vis_processor=vis_processor,
                text_processor=text_processor,
                audio_processor=audio_processor,
                audio_dir=audio_path,
                ann_path=eval_file_path,
                image_root=img_path,
                swap_order=swap_order
            )
        elif vis_processor_name == 'videomae_processor':
            data = VideoDataset(
                vis_processor=vis_processor,
                text_processor=text_processor,
                audio_processor=audio_processor,
                audio_dir=audio_path,
                ann_path=eval_file_path,
                video_root=img_path
            )
        else:
            raise RuntimeError(f"Can not find suitable vision processor for the vision input type!")
        
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        results = []
        for batch in tqdm(eval_dataloader):
            # logging.info(f"evaluate: batch keys: {list(batch.keys())}, audio present: {'audio' in batch}")
            images = batch["image"].half()
            audios = batch.get("audio", None)
            instruction_input = batch["instruction_input"]
            ground_truth = batch["answer"]
            text_questions = batch["question"]
            image_ids = batch["image_id"]
            texts = prepare_texts(instruction_input, conv_temp)
            predicts = model.generate(images=images,
                                      audios=audios,
                                      texts=texts,
                                      max_new_tokens=max_new_tokens,
                                      temperature=temperature,
                                      top_p=top_p,
                                      do_sample=do_sample)

            results.extend([{
                "image_id": image_id,
                "text_question": text_question,
                "ground_truth": gt,
                "predict": predict
            } for image_id, text_question, gt, predict in zip(image_ids, text_questions, ground_truth, predicts)])
            logging.info(f"evaluate: predicts: {predicts[:2]}")
            # break

        # Save outputs and metrics into separate subfolders under --output-dir.
        outputs_dir = os.path.join(args.output_dir, "outputs")
        metrics_dir = os.path.join(args.output_dir, "metrics")
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        save_path = os.path.join(
            outputs_dir, f"output_{dataset_name}_{reasoning_string}_{swap_order_string}{pretrained_string}.json"
        )

        with open(save_path, "w") as jsonfile:
            json.dump(results, jsonfile, ensure_ascii=False)

        logging.info(f"Saving results to: {save_path}")

        # Compute and save metrics next to the output JSON.
        try:
            accuracy = calculate_accuracy(results)
            bleu = calculate_bleu(results)
            rouge_scores = calculate_rouge(results)

            metrics = {
                "accuracy": accuracy,
                "bleu": bleu,
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
            }

            metrics_path = os.path.join(
                metrics_dir,
                f"metrics_output_{dataset_name}_{reasoning_string}_{swap_order_string}{pretrained_string}.json",
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logging.info(f"Metrics saved to: {metrics_path}")
        except Exception as e:
            logging.exception(f"Failed to compute/save metrics for {save_path}: {e}")

if __name__ == "__main__":
    args = parse_args()
    logging.info("Evaluating...")
    evaluate(args)
    logging.info("Done!")