from src.tokenizer import BytePairTokenizer, load_tokenizer

def main() -> None:
    tokenizer = load_tokenizer()
    text = 'Sean Bean has a hard time leaving his role as Eddard Stark . He vows to get revenge against those that assisted in his execution , starting with George R. R. Martin'
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {len(tokenizer.token_map)}")

if __name__ == "__main__":
    main()