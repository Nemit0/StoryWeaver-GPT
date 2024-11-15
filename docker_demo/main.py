import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.count_primes import count_primes

def main() -> None:
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")
    # Perform large scale computation
    x = torch.rand(1000, 1000)
    y = torch.rand(1000, 1000)
    z = torch.matmul(x, y)
    print(z)

    # Count the number of prime numbers up to 1 million
    N = 10000
    x = range(N)
    y = [0] * N
    for i in tqdm(range(N)):
        y[i] = count_primes(i)
    
    plt.plot(x, y)
    plt.xlabel("N")
    plt.ylabel("Primes")
    plt.title("Prime distribution")
    plt.savefig("primes.png")

if __name__ == "__main__":
    main()