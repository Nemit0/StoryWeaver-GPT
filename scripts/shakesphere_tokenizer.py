import os

from typing import List

try:
    from rich import print
except ImportError:
    print("rich not found. Defaulting to print")

from tokenizer import BytePairTokenizer

def main() -> None:
    # Prepare to load data
    print(os.getcwd())
    project_root = os.getcwd()
    print(f"Project Root: {project_root}")
    data_path:str = os.path.join(project_root, 'input.txt')
    # Load the data
    with open(data_path, 'r', encoding='utf-8') as file:
        data:str = file.read()
    
    print(f"Data length: {len(data)}")
    
    data_list = [data]

    print(f"Data list length: {len(data_list)}")

    # Initialize the tokenizer
    tokenizer = BytePairTokenizer()
    num_merges:int = 1024
    tokenizer.train(data_list, num_merges=num_merges, verbose=True)

    # Save the model
    model_path:str = os.path.join(project_root, 'model', 'tokenizer_shakesphere.json')
    tokenizer.save_model(model_path)
    print(f"Model saved to {model_path}")
    del tokenizer

    # Load the model
    tokenizer = BytePairTokenizer()
    tokenizer.load_model(model_path)

    # Test the tokenizer
    text = ' low lower newest widest'
    encoded = tokenizer.encode(text)

    print(f"Encoded: {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    return 0

if __name__ == '__main__':
    main()