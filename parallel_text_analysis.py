import time
from multiprocessing import Pool
from collections import Counter
import pandas as pd
import csv
import string

# ---------------------------
# Load reviews from CSV
def load_reviews(file_path='IMDB Dataset.csv'):
    df = pd.read_csv(file_path)
    # Assuming the column containing reviews is named 'review'
    return df['review'].tolist()

# ---------------------------
# Preprocess and tokenize text
def tokenize(text):
    # Lowercase, remove punctuation, split by space
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return words

# ---------------------------
# Function to process a chunk of data
def process_chunk(chunk):
    counter = Counter()
    for text in chunk:
        words = tokenize(text)
        counter.update(words)
    return counter

# ---------------------------
# Function to split data into chunks
def split_data(data, n_chunks):
    avg = len(data) // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * avg
        end = None if i == n_chunks - 1 else (i + 1) * avg
        chunks.append(data[start:end])
    return chunks

# ---------------------------
# Run parallel analysis
def run_parallel_analysis(reviews, num_workers_list=[1, 2, 4, 8]):
    print(f"Running parallel text analysis on {len(reviews)} reviews...\n")
    
    times = []

    for workers in num_workers_list:
        start_time = time.time()
        
        if workers == 1:
            # Sequential
            result_counter = process_chunk(reviews)
        else:
            # Parallel
            chunks = split_data(reviews, workers)
            with Pool(processes=workers) as pool:
                results = pool.map(process_chunk, chunks)
            
            # Reduction: merge all counters
            result_counter = Counter()
            for r in results:
                result_counter.update(r)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"Workers= {workers} | Time={elapsed:.2f}s")
    
    # Compute Speedup and Efficiency
    print("\nSummary table:")
    print("Workers | Time (s) | Speedup | Efficiency (%)")
    print("----------------------------------------------")
    t1 = times[0]
    for i, t in enumerate(times):
        workers = num_workers_list[i]
        speedup = t1 / t
        efficiency = (speedup / workers) * 100
        print(f"{workers:7} | {t:8.2f} | {speedup:6.2f} | {efficiency:12.1f}")
    
    # Save top 20 words
    top_words = result_counter.most_common(20)
    with open('para_output.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Word', 'Count'])
        writer.writerows(top_words)
    
    print("\nTop 20 words saved to para_output.csv")
    print("Top words:")
    print(", ".join([f"{w}({c})" for w, c in top_words]))

# ---------------------------
# Main execution
if __name__ == "__main__":
    reviews = load_reviews(r'D:\6th Semester\PDC-MidLab\IMDB Dataset.csv')
    run_parallel_analysis(reviews)
