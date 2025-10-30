from mpi4py import MPI
from collections import Counter
import pandas as pd
import string
import csv

# ---------------------------
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------------------
# Load dataset on master (rank 0)
reviews = []
if rank == 0:
    df = pd.read_csv('IMDB Dataset.csv')
    reviews = df['review'].tolist()

# ---------------------------
# Optional imbalance: rank 0 gets more data
if rank == 0:
    if size > 1:
        # Rank 0 gets half
        chunks = [reviews[:len(reviews)//2]]
        remaining = reviews[len(reviews)//2:]
        chunk_size = len(remaining) // (size - 1)
        for i in range(1, size):
            start = (i-1) * chunk_size
            end = None if i == size-1 else i*chunk_size
            chunks.append(remaining[start:end])
    else:
        # Only one process
        chunks = [reviews]
else:
    chunks = None

# ---------------------------
# Scatter chunks to all ranks
my_chunk = comm.scatter(chunks, root=0)

# ---------------------------
# Tokenizer and word count function
def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

# ---------------------------
# Process chunk and measure time
start_time = MPI.Wtime()

my_counter = Counter()
for text in my_chunk:
    words = tokenize(text)
    my_counter.update(words)

end_time = MPI.Wtime()
elapsed = end_time - start_time

# ---------------------------
# Gather results at master
all_counters = comm.gather(my_counter, root=0)
all_times = comm.gather(elapsed, root=0)

# ---------------------------
# Master aggregates results
if rank == 0:
    total_counter = Counter()
    for c in all_counters:
        total_counter.update(c)

    print("\nPer rank processing times:")
    for i, t in enumerate(all_times):
        print(f"Rank {i} processed {len(chunks[i])} lines in {t:.2f}s")
    
    print(f"\nTotal distributed time: {max(all_times):.2f}s")

    # Simple speedup estimate over sequential (sum of times)
    seq_time = sum(all_times)
    speedup = seq_time / max(all_times)
    print(f"Estimated speedup over sequential: {speedup:.2f}x")

    # Top 20 words
    top_words = total_counter.most_common(20)
    print("\nTop 20 words:")
    print(", ".join([f"{w}({c})" for w, c in top_words]))

    # Save top words to CSV
    with open('mpi_top_words.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Word', 'Count'])
        writer.writerows(top_words)
    print("\nTop 20 words saved to mpi_top_words.csv")
