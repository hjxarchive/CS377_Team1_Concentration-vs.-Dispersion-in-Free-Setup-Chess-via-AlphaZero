import numpy as np
import time

def current_key_fn(arr: np.ndarray) -> str:
    return bytes(arr[arr > 0].tolist()).decode("ascii")

def current_decode_fn(arr: np.ndarray) -> tuple[str, list[str]]:
    text = bytes(arr[arr > 0].tolist()).decode("ascii")
    parts = text.split("\n", 1)
    start_fen = parts[0]
    if len(parts) > 1 and parts[1].strip():
        moves = parts[1].strip().split()
    else:
        moves = []
    return start_fen, moves

def optimized_key_fn(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def optimized_decode_fn(arr: np.ndarray) -> tuple[str, list[str]]:
    # Find first zero to avoid decoding the whole padded array
    zero_indices = np.where(arr == 0)[0]
    length = zero_indices[0] if len(zero_indices) > 0 else len(arr)
    text = arr[:length].tobytes().decode("ascii")
    parts = text.split("\n", 1)
    start_fen = parts[0]
    if len(parts) > 1 and parts[1].strip():
        moves = parts[1].strip().split()
    else:
        moves = []
    return start_fen, moves

# Create a sample array representing standard FEN + a few moves
start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
text = start_fen + "\n" + " ".join(moves)
text_bytes = text.encode("ascii")

arr = np.zeros(4096, dtype=np.uint8)
arr[:len(text_bytes)] = list(text_bytes)

# Warmup
for _ in range(100):
    current_key_fn(arr)
    current_decode_fn(arr)
    optimized_key_fn(arr)
    optimized_decode_fn(arr)

# Benchmark current key generation
n_iter = 100000

t0 = time.time()
for _ in range(n_iter):
    current_key_fn(arr)
t1 = time.time()
current_key_time = t1 - t0

t0 = time.time()
for _ in range(n_iter):
    optimized_key_fn(arr)
t1 = time.time()
optimized_key_time = t1 - t0

print(f"Key generation ({n_iter} iterations):")
print(f"  Current method:   {current_key_time:.4f}s")
print(f"  Optimized method: {optimized_key_time:.4f}s")
print(f"  Speedup:          {current_key_time / optimized_key_time:.1f}x")

# Benchmark decode function
t0 = time.time()
for _ in range(n_iter):
    current_decode_fn(arr)
t1 = time.time()
current_decode_time = t1 - t0

t0 = time.time()
for _ in range(n_iter):
    optimized_decode_fn(arr)
t1 = time.time()
optimized_decode_time = t1 - t0

print(f"\nDecode function ({n_iter} iterations):")
print(f"  Current method:   {current_decode_time:.4f}s")
print(f"  Optimized method: {optimized_decode_time:.4f}s")
print(f"  Speedup:          {current_decode_time / optimized_decode_time:.1f}x")
