import json
import re
import os
from typing import List, Dict, Tuple
from multiprocessing import Pool
from tqdm import tqdm

class BytePairTokenizer:
    def __init__(self, data_path:str=None) -> None:
        """
        BytePairTokenizer object
        """
        if data_path:
            self.load_model(data_path)
            return
        
        self.special_tokens:Dict[str, int] = {
            '<BOT>': 0,  # Beginning of Text
            '<EOT>': 1,   # End of Text
            '</w>': 2     # end of word
        }
        self.inv_special_tokens:Dict[int, str] = {i: t for t, i in self.special_tokens.items()}

        self.token_map: Dict[str, int] = self.special_tokens.copy()
        self.inv_map: Dict[int, str] = self.inv_special_tokens.copy()
        self.bpe_codes: Dict[Tuple[str, str], int] = {}
    
    def train(self, corpus: List[str], num_merges: int, verbose:bool = False) -> None:
        """
        Train the Byte Pair Tokenizer to process sentences.
        """
        # Build the vocabulary: map token sequences to their frequencies
        vocab = {}
        if verbose:
            print("Building vocabulary...")
        for sentence in tqdm(corpus):
            # Split sentence into words with leading whitespace preserved
            words = re.findall(r'\s*\S+|\s+', sentence)
            for word in words:
                # Skip special tokens
                if word in self.special_tokens.keys():
                    continue
                chars = list(word) + ['</w>']
                word_tuple = tuple(chars)
                vocab[word_tuple] = vocab.get(word_tuple, 0) + 1
        
        if verbose:
            print("Vocabulary built.\nTraining BPE...")
        token_id = len(self.token_map)  # Starting token ID
        symbols = set()
        for word_tuple in vocab.keys():
            symbols.update(word_tuple)
        for symbol in symbols:
            if symbol not in self.token_map:
                self.token_map[symbol] = token_id
                token_id += 1
        self.inv_map = {i: t for t, i in self.token_map.items()}
        
        if verbose:
            print("Token map built.\nMerging tokens...")
        # Perform BPE merges
        for i in tqdm(range(num_merges)):
            pairs = self._get_pair_counts(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.bpe_codes[best_pair] = i # Record the BPE merge rule
            new_symbol = ''.join(best_pair)
            if new_symbol not in self.token_map:
                self.token_map[new_symbol] = token_id
                token_id += 1
                self.inv_map[self.token_map[new_symbol]] = new_symbol
    
    def _get_pair_counts(self, vocab: Dict[Tuple[str], int]) -> Dict[Tuple[str, str], int]:
        """
        Get counts of symbol pairs in the vocabulary
        """
        pairs = {}
        for word, freq in vocab.items():
            symbols = word
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs
    
    def _merge_vocab_single(self, pair: Tuple[str, str], vocab: Dict[Tuple[str], int]) -> Dict[Tuple[str], int]:
        """
        Merge all occurrences of the given pair in the vocabulary
        """
        new_vocab = {}
        bigram = ''.join(pair)
        for word, freq in vocab.items():
            w = []
            i = 0
            while i < len(word):
                # Merge the pair if found
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    w.append(bigram)
                    i += 2
                else:
                    w.append(word[i])
                    i += 1
            new_vocab[tuple(w)] = freq
        return new_vocab

    @staticmethod
    def _process_word(args):
        pair, word_freq = args
        word, freq = word_freq
        bigram = ''.join(pair)
        w = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                w.append(bigram)
                i += 2
            else:
                w.append(word[i])
                i += 1
        return tuple(w), freq
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[Tuple[str], int]) -> Dict[Tuple[str], int]:
        """
        Parallel merge of all occurrences of the given pair in the vocabulary using multiprocessing.
        """
        with Pool() as pool:
            results = pool.map(self._process_word, [(pair, word_freq) for word_freq in vocab.items()])

        new_vocab = {word: freq for word, freq in results}
        return new_vocab
    
    def _get_pairs(self, word: List[str]) -> set:
        """
        Return a set of symbol pairs in a word
        """
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs
    
    def _apply_bpe(self, word: List[str]) -> List[str]:
        """
        Apply BPE to a list of symbols (a word)
        """
        word = word.copy()
        pairs = self._get_pairs(word)
        while True:
            if not pairs:
                break
            # Find the highest priority pair to merge
            min_pair = None
            min_rank = float('inf')
            for pair in pairs:
                if pair in self.bpe_codes:
                    rank = self.bpe_codes[pair]
                    if rank < min_rank:
                        min_rank = rank
                        min_pair = pair
            if min_pair is None:
                break
            # Merge the best pair
            new_symbol = ''.join(min_pair)
            i = 0
            while i < len(word) - 1:
                if word[i] == min_pair[0] and word[i + 1] == min_pair[1]:
                    word[i:i + 2] = [new_symbol]
                    i = max(i - 1, 0)  # Restart from the previous position after a merge
                else:
                    i += 1
            pairs = self._get_pairs(word)
        return word
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into BPE tokens with leading whitespace preserved
        """
        tokens = []
        words = re.findall(r'\s*\S+|\s+', text)
        for word in words:
            chars = list(word) + ['</w>']
            bpe_word = self._apply_bpe(chars)
            tokens.extend(bpe_word)
        return tokens
    
    def encode(self, data: str) -> List[int]:
        """
        Encode text data into a list of token IDs
        """
        str_list = self.split_text(data)
        token_list = [self.token_map[tok] for tok in str_list]
        return token_list
    
    def decode(self, data: List[int]) -> str:
        """
        Decode a list of token IDs back into text
        """
        tokens = [self.inv_map[i] for i in data]
        text = ''
        for token in tokens:
            if token != '</w>':
                text += token.replace('</w>', '')
        return text

    def save_model(self, target_path:str) -> None:
        """
        Save the model to a file as json file
        the json will look like
        {
            token_map : {...},
            bpe_codes : {...}
        }
        The special tokens are not necessary for simple encoding/decoding
        hence it is omitted from the model
        """
        with open(target_path, 'w', encoding="UTF-8") as f:
            json.dump({
                'token_map': self.token_map,
                'bpe_codes': {json.dumps(list(k)): v for k, v in self.bpe_codes.items()}
            }, f,
             indent=4,
              ensure_ascii=False)
    
    def load_model(self, model_path:str, encoding="UTF-8") -> None:
        """
        Load the model from a json file
        JSON doesn't allow tuple object as key
        hence the tuple keys are converted to string before saving
        and converted back to tuple when loading
        """
        with open(model_path, 'r') as f:
            model = json.load(f)
        self.token_map = model['token_map']
        self.inv_map = {i: t for t, i in self.token_map.items()}
        self.bpe_codes = {tuple(json.loads(k)): v for k, v in model['bpe_codes'].items()}

def load_tokenizer(path:str = None) -> BytePairTokenizer:
    """
    Load the BytePairTokenizer model from the model folder
    """
    if path is None:
        model_path:str = os.path.join(os.getcwd(), 'model', 'tokenizer.json')
    else:
        model_path:str = path
    tokenizer = BytePairTokenizer(model_path)
    # tokenizer.load_model(model_path)
    return tokenizer