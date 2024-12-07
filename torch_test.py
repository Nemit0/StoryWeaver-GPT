import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable

if torch.cuda.is_available():
    print("CUDA is available")
    torch.set_default_device('cuda')
else:
    print("CUDA is not available")
    torch.set_default_device('cpu')

def timeit(func: Callable) -> Callable:
    """
    Decorator function to measure the execution time of a function.

    Args:
        func (Callable): The function to measure the execution time.

    Returns:
        Callable: The wrapper function that measures the execution time.
    """
    import time

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.6f} seconds")
        return result

    return wrapper

def count_primes(N:int) -> int:
    """
    Counts the number of prime numbers up to N using PyTorch and CUDA.

    Args:
        N (int): The upper limit of the range to check for prime numbers.

    Returns:
        int: The count of prime numbers up to N.
    """
    # Ensure N is at least 2
    if N < 2:
        return 0

    # Set the device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize a boolean tensor where index represents the number
    sieve = torch.ones(N + 1, dtype=torch.bool, device=device)
    sieve[0:2] = False  # 0 and 1 are not prime numbers

    # Implement the Sieve of Eratosthenes
    max_limit = int(N ** 0.5) + 1
    for p in range(2, max_limit):
        if sieve[p]:
            # Mark multiples of the prime number as non-prime
            sieve[p*p:N+1:p] = False

    # Count the number of primes
    prime_count = sieve.sum().item()

    return prime_count

def main() -> None:
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