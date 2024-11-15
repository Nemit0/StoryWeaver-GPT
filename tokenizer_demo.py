from src.tokenizer import BytePairTokenizer, load_tokenizer

def main() -> None:
    tokenizer = load_tokenizer()
    text = ' low lower newest widest'
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

if __name__ == "__main__":
    main()