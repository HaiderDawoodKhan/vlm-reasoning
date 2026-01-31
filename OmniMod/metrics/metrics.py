import json
import string
import os
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

def parse_args():
    parser = argparse.ArgumentParser(description="Compute text metrics from an evaluation JSON.")
    parser.add_argument(
        "--file-path",
        required=True,
        type=str,
        help="Path to the JSON predictions file (list of dicts with 'predict' and 'ground_truth').",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Directory to write metrics_output.json (defaults to the directory of --file-path).",
    )
    return parser.parse_args()


def ensure_nltk_data():
    """Ensure required NLTK tokenizers are available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

def normalize_text(text):
    """Normalize text by converting to lowercase and removing punctuation."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def load_data(file_path):
    """Load JSON data from the specified file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_accuracy(data):
    """Calculate accuracy as the fraction of exact matches after normalization."""
    correct = sum(1 for item in data if normalize_text(item['predict']) == normalize_text(item['ground_truth']))
    total = len(data)
    return correct / total if total > 0 else 0.0

def calculate_bleu(data):
    """Calculate average BLEU score for predictions after normalization."""
    smoothie = SmoothingFunction().method4  
    bleu_scores = []
    for item in data:
        reference = nltk.word_tokenize(normalize_text(item['ground_truth']))
        hypothesis = nltk.word_tokenize(normalize_text(item['predict']))
        score = sentence_bleu([reference], hypothesis, smoothing_function=smoothie)
        bleu_scores.append(score)
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

def calculate_rouge(data):
    """Calculate average ROUGE-1, ROUGE-2, and ROUGE-L scores after normalization."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    
    for item in data:
        norm_ground_truth = normalize_text(item['ground_truth'])
        norm_predict = normalize_text(item['predict'])
        scores = scorer.score(norm_ground_truth, norm_predict)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
    }

def save_metrics(metrics, output_dir):
    """Save metrics to a JSON file in the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'metrics_output.json')
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {output_path}")

def main():
    args = parse_args()
    ensure_nltk_data()

    file_path = args.file_path
    output_dir = args.output_dir or os.path.dirname(file_path)
    
    # Load data
    data = load_data(file_path)
    
    # Calculate metrics
    accuracy = calculate_accuracy(data)
    bleu = calculate_bleu(data)
    rouge_scores = calculate_rouge(data)
    
    # Compile metrics
    metrics = {
        'accuracy': accuracy,
        'bleu': bleu,
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL']
    }
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"BLEU Score: {bleu:.4f}")
    print(f"ROUGE-1 F1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2 F1: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L F1: {rouge_scores['rougeL']:.4f}")
    
    # Save metrics to JSON
    save_metrics(metrics, output_dir)

if __name__ == "__main__":
    main()