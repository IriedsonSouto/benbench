from ppl_and_ngram_utils import *
import argparse
import os
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Benchmark Leakage Detection based on PPL', add_help=False)
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset category (gsm8k, math, enem_challenge)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--device', type=str, required=True, help='Device (e.g., cpu, cuda)')
    args = parser.parse_args()

    # Carregar o modelo
    model, tokenizer = load_model(args.model_path, args.device)

    # Determinar quais datasets serão usados
    if args.dataset_name == "gsm8k":
        dataset_names = [
            "GSM8K_rewritten-test-1",
            "GSM8K_rewritten-test-2",
            "GSM8K_rewritten-test-3",
            "GSM8K_rewritten-train-1",
            "GSM8K_rewritten-train-2",
            "GSM8K_rewritten-train-3",
            "orgn-GSM8K-test",
            "orgn-GSM8K-train",
        ]
    elif args.dataset_name == "math":
        dataset_names = [
            "MATH_rewritten-test-1",
            "MATH_rewritten-test-2",
            "MATH_rewritten-test-3",
            "MATH_rewritten-train-1",
            "MATH_rewritten-train-2",
            "MATH_rewritten-train-3",
            "orgn-MATH-train",
            "orgn-MATH-test",
        ]
    elif args.dataset_name == "enem_challenge":
        # Procurar dinamicamente todos os arquivos "enem_challenge"
        dataset_names = ["orgn-enem_challenge"] + [
            f.partition("-")[2].replace(".jsonl", "")
            for f in os.listdir("./data/partition-enem_challenge")
            if f.startswith("partition-enem_challenge")
        ]
    else:
        raise ValueError("Invalid dataset name. Use 'gsm8k', 'math', or 'enem_challenge'.")

    results_ppl_summary = {}

    for dataset_name in dataset_names:
        # Determinar o caminho do dataset
        if "rewritten" in dataset_name:
            dataset_path = f'./data/rewritten/{dataset_name}.jsonl'
        elif "orgn" in dataset_name:
            dataset_path = f'./data/original/{dataset_name}.jsonl'
        elif "enem_challenge" in dataset_name:
            # Arquivo geral está em "orgn", partições estão em "partition-enem_challenge"
            dataset_path = (
                f'./data/orgn/{dataset_name}.jsonl'
                if dataset_name == "orgn-enem_challenge"
                else f'./data/partition-enem_challenge/{dataset_name}.jsonl'
            )
        else:
            raise ValueError(f"Unexpected dataset name: {dataset_name}")

        # Validar se o arquivo existe
        if not os.path.exists(dataset_path):
            print(f"Dataset file not found: {dataset_path}")
            continue

        # Carregar os dados do JSONL
        dataset = load_data_from_jsonl(dataset_path)

        # Definir o caminho do arquivo de saída
        output_file_ppl = f'./outputs/ppl/ppl-{args.model_name}-{dataset_name}.jsonl'

        # Calcular a perplexidade (PPL)
        ppl_results = calculate_answer_ppl(dataset, model, tokenizer, args.device, output_file_ppl)
        print(f"{dataset_name} Average_p
