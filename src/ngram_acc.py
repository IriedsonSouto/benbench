from ppl_and_ngram_utils import *
import argparse
import os
import re

# Função para determinar o caminho do dataset
def get_dataset_path(dataset_name):
    """
    Determina o caminho do dataset com base no nome fornecido e extrai o ano, se disponível.
    """
    # Se for o arquivo geral
    if dataset_name == "orgn-enem_challenge":
        return f'./data/original/{dataset_name}.jsonl'

    # Extrair o ano do nome do arquivo, se existir
    match = re.search(r"partition-enem_challenge_(\d{4})", dataset_name)
    if match:
        year = match.group(1)
        partition_file = f'./data/partition-enem_challenge/{dataset_name}.jsonl'
        if os.path.exists(partition_file):
            return partition_file
        else:
            raise FileNotFoundError(f"Partition file for year {year} not found: {partition_file}")

    raise ValueError(f"Invalid dataset name format: {dataset_name}")

# Função principal de processamento
def process_dataset(dataset_path, n, k, model, tokenizer, device, model_name, model_type):
    """
    Processa o dataset e calcula a acurácia do n-grama.
    """
    # Carregar os dados
    dataset = load_data_from_jsonl(dataset_path)

    # Determinar nome base para saída
    dataset_name = os.path.basename(dataset_path).replace(".jsonl", "")
    output_file = f'./outputs/ngram/{n}gram-{model_name}-{dataset_name}.jsonl'

    # Calcular a acurácia do n-grama
    ngram_results = calculate_n_gram_accuracy(n, k, dataset, model, tokenizer, device, output_file, model_type)
    print(f"{dataset_name} {n}_gram_accuracy: {ngram_results['mean_n_grams']}")
    return {dataset_name: ngram_results["mean_n_grams"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Benchmark Leakage Detection based on PPL', add_help=False)
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset file name (e.g., partition-enem_challenge_2020)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--model_type', type=str, default="base", help='Model type: base or chat')
    parser.add_argument('--device', type=str, required=True, help='Device (e.g., cpu, cuda)')
    parser.add_argument('--n', type=int, default=5, help='N-gram size (default: 5)')
    parser.add_argument('--k', type=int, default=5, help='Number of starting points (default: 5)')
    args = parser.parse_args()

    # Carregar modelo e tokenizer
    model, tokenizer = load_model(args.model_path, args.device)

    # Determinar caminho do dataset com base no argumento
    dataset_path = get_dataset_path(args.dataset_name)

    # Processar o dataset
    results_summary = process_dataset(
        dataset_path,
        args.n,
        args.k,
        model,
        tokenizer,
        args.device,
        args.model_name,
        args.model_type,
    )

    # Exibir o resumo dos resultados
    print(f"\nN-gram accuracy for {args.model_name}:")
    for dataset, accuracy in results_summary.items():
        print(f"{dataset}: {accuracy}")
