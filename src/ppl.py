from ppl_and_ngram_utils import *
import argparse

# Dataset names
gsm8k_dataset_names = [
    "GSM8K_rewritten-test-1",
    "GSM8K_rewritten-test-2",
    "GSM8K_rewritten-test-3",
    "GSM8K_rewritten-train-1",
    "GSM8K_rewritten-train-2",
    "GSM8K_rewritten-train-3",
    "orgn-GSM8K-test",
    "orgn-GSM8K-train",
]
math_dataset_names = [
    "MATH_rewritten-test-1",
    "MATH_rewritten-test-2",
    "MATH_rewritten-test-3",
    "MATH_rewritten-train-1",
    "MATH_rewritten-train-2",
    "MATH_rewritten-train-3",
    "orgn-MATH-train",
    "orgn-MATH-test",
]
enem_challenge_dataset_names = [
    "enem_challenge-test-rewritten-1",
    "enem_challenge-test-rewritten-2",
    "enem_challenge-test-rewritten-3",
    "enem_challenge-train-rewritten-1",
    "enem_challenge-train-rewritten-2",
    "enem_challenge-train-rewritten-3",
    "orgn-enem_challenge-train",
    "orgn-enem_challenge-test",
]

if __name__ == "__main__":
    # Script arguments
    parser = argparse.ArgumentParser('Benchmark Leakage Detection based on Perplexity (PPL)', add_help=False)
    parser.add_argument('--dataset_name', type=str, required=True, 
                        help='Dataset category (gsm8k, math, enem_challenge)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--device', type=str, required=True, help='Device (e.g., cpu, cuda)')
    args = parser.parse_args()

    # Load the model
    model, tokenizer = load_model(args.model_path, args.device)

    # Determine which datasets to use
    if args.dataset_name == "gsm8k":
        dataset_names = gsm8k_dataset_names
    elif args.dataset_name == "math":
        dataset_names = math_dataset_names
    elif args.dataset_name == "enem_challenge":
        dataset_names = enem_challenge_dataset_names
    else:
        raise ValueError("Invalid dataset name. Use 'gsm8k', 'math', or 'enem_challenge'.")

    # Store results
    results_ppl_summary = {}

    # Iterate over datasets
    for dataset_name in dataset_names:
        # Determine dataset path
        if "rewritten" in dataset_name:
            dataset_path = f'./data/rewritten/{dataset_name}.jsonl'
        elif "orgn" in dataset_name:
            dataset_path = f'./data/original/{dataset_name}.jsonl'
        else:
            raise ValueError(f"Unexpected dataset name: {dataset_name}")

        # Load dataset
        dataset = load_data_from_jsonl(dataset_path)

        # Skip if dataset is empty
        if not dataset:
            print(f"Dataset {dataset_name} is empty. Skipping...")
            continue

        # Output file for PPL results
        output_file_ppl = f'./outputs/ppl/ppl-{args.model_name}-{dataset_name}.jsonl'

        # Calculate perplexity
        ppl_results = calculate_answer_ppl(dataset, model, tokenizer, args.device, output_file_ppl)
        print(f"{dataset_name} Average PPL Accuracy: ", ppl_results["mean_perplexity"])
        results_ppl_summary[f'{dataset_name}'] = ppl_results["mean_perplexity"]

    # Print summary of results
    print(f"PPL Results for {args.model_name}")
    for key, value in results_ppl_summary.items():
        print(f"{key}: {value}")
