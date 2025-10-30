from mpi4py import MPI
import pandas as pd
import time
import string
from collections import Counter
from multiprocessing import Pool
import numpy as np

# Load stopwords
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most', 
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'br'
])

def clean_text(text):
    """Clean text by lowercasing, removing punctuation and stopwords"""
    text = text.lower()
    text = text.replace('<br />', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word and word not in STOPWORDS]
    return words

def process_subchunk(reviews):
    """Process a subchunk of reviews (used by multiprocessing)"""
    local_counter = Counter()
    for text in reviews:
        words = clean_text(str(text))
        local_counter.update(words)
    return local_counter

def hybrid_process(data_chunk, num_threads=4):
    """
    Process a chunk using multiprocessing
    data_chunk: tuple of (reviews, sentiment_filter)
    sentiment_filter: 'positive', 'negative', or None
    """
    reviews, sentiments, sentiment_filter = data_chunk
    
    # Filter by sentiment if specified
    if sentiment_filter:
        filtered_data = []
        for review, sentiment in zip(reviews, sentiments):
            if sentiment_filter.lower() in sentiment.lower():
                filtered_data.append(review)
        reviews = filtered_data
    
    if len(reviews) == 0:
        return Counter()
    
    # Split into subchunks for local parallel processing
    subchunks = np.array_split(reviews, num_threads)
    
    # Process using multiprocessing
    with Pool(processes=num_threads) as pool:
        results = pool.map(process_subchunk, subchunks)
    
    # Combine local results
    final_counter = Counter()
    for counter in results:
        final_counter.update(counter)
    
    return dict(final_counter)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    num_local_threads = 4  # Number of threads per MPI process
    
    overall_start = time.perf_counter()
    
    if rank == 0:
        # Master: Load and distribute data
        print(f"Hybrid Parallel NLP System")
        print(f"MPI Processes: {size}, Threads per process: {num_local_threads}")
        print("=" * 60)
        print("Loading dataset...")
        
        df = pd.read_csv('IMDB Dataset.csv', nrows=20000)
        reviews = df['review'].tolist()
        sentiments = df['sentiment'].tolist()
        
        print(f"Loaded {len(reviews)} reviews")
        
        # Create chunks for each MPI process
        chunks = []
        if size >= 2:
            # Assign sentiment-specific tasks
            # Rank 0: positive reviews
            # Rank 1: negative reviews
            # Other ranks: all reviews
            for i in range(size):
                if i == 0:
                    chunks.append((reviews, sentiments, 'positive'))
                elif i == 1:
                    chunks.append((reviews, sentiments, 'negative'))
                else:
                    # Split remaining data
                    start_idx = (i - 2) * len(reviews) // (size - 2 if size > 2 else 1)
                    end_idx = (i - 1) * len(reviews) // (size - 2 if size > 2 else 1)
                    chunks.append((reviews[start_idx:end_idx], sentiments[start_idx:end_idx], None))
        else:
            chunks.append((reviews, sentiments, None))
        
        print(f"Data distributed among {size} nodes\n")
    else:
        chunks = None
    
    # Scatter data
    local_data = comm.scatter(chunks, root=0)
    
    # Process locally with hybrid parallelism
    local_start = time.perf_counter()
    local_counter = hybrid_process(local_data, num_local_threads)
    local_end = time.perf_counter()
    local_time = local_end - local_start
    
    # Determine label for this rank
    if rank == 0 and size >= 2:
        label = "positive"
    elif rank == 1 and size >= 2:
        label = "negative"
    else:
        label = "all"
    
    num_processed = len(local_data[0]) if local_data[2] is None else sum(1 for s in local_data[1] if local_data[2].lower() in s.lower())
    
    print(f"Node {rank} ({label}): processed {num_processed} reviews in {local_time:.1f}s")
    
    # Gather results
    comm_start = time.perf_counter()
    all_counters = comm.gather(local_counter, root=0)
    comm_end = time.perf_counter()
    
    if rank == 0:
        comm_overhead = comm_end - comm_start
        
        # Aggregate all counters
        print("\nAggregating results...")
        final_counter = Counter()
        for counter_dict in all_counters:
            final_counter.update(counter_dict)
        
        overall_end = time.perf_counter()
        total_time = overall_end - overall_start
        
        # Get top words
        top_20 = final_counter.most_common(20)
        
        print(f"\nCommunication overhead: {comm_overhead:.1f}s")
        print(f"Hybrid total time: {total_time:.1f}s")
        
        # Calculate speedup (update with your sequential baseline)
        sequential_time = 58.0
        speedup = sequential_time / total_time
        print(f"Hybrid speedup: {speedup:.1f}x vs Sequential")
        
        print(f"\nTop 20 most common words (across all sentiments):")
        for i, (word, freq) in enumerate(top_20[:10], 1):
            print(f"  {i}. {word}: {freq}")
        
        # Show sentiment-specific top words if applicable
        if size >= 2:
            print(f"\nSentiment-specific analysis:")
            positive_counter = Counter(all_counters[0])
            negative_counter = Counter(all_counters[1])
            
            print(f"\nTop 5 positive sentiment words:")
            for word, freq in positive_counter.most_common(5):
                print(f"  {word}: {freq}")
            
            print(f"\nTop 5 negative sentiment words:")
            for word, freq in negative_counter.most_common(5):
                print(f"  {word}: {freq}")

if __name__ == "__main__":
    main()
