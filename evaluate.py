import os
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

def evaluate(args):
    cfg = Config(args)
    
    reasoning_string = "no_reasoning"
    if args.coconut:
        cfg.model_cfg.use_coconut = True
        reasoning_string = "coconut"
    else:
        cfg.model_cfg.use_coconut = False
        
    if args.multinut:
        cfg.model_cfg.use_multimodal_coconut = True
        reasoning_string = "multinut"
    else:
        cfg.model_cfg.use_multimodal_coconut = False
    
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
            outputs_dir, f"output_{dataset_name}_{reasoning_string}_{swap_order_string}.json"
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
                f"metrics_output_{dataset_name}_{reasoning_string}_{swap_order_string}.json",
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