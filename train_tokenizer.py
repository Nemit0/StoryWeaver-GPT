import os

from typing import List

try:
    from rich import print
except ImportError:
    print("rich not found. Defaulting to print")

from src.tokenizer import BytePairTokenizer
from src.utils import get_project_root

def main() -> None:
    # Prepare to load data
    project_root:str = get_project_root('mlgroup1')
    print(f"Project Root: {project_root}")
    data_path:str = os.path.join(project_root, 'data')
    source_list:List[str] = os.listdir(data_path)    
    if not source_list:
        raise ValueError("No dataset found in the data folder\nDataset should be in /mlgroup1/data")
    
    source_list.remove('README')
    print(f"Datasets found: {source_list}")

    # Load the data
    data:str = ''
    for file in source_list:
        with open(os.path.join(data_path, file), 'r') as f:
            data += f.read()
    
    print(f"Data length: {len(data)}")

    data.replace('\n', '<newline>')

    # Initialize the tokenizer
    tokenizer = BytePairTokenizer()
    num_merges:int = 4
    tokenizer.train(data, num_merges=num_merges)

    # Save the model
    model_path:str = os.path.join(project_root, 'models', 'tokenizer.json')
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