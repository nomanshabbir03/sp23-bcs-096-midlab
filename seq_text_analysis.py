import pandas as pd
import re
import string
import time
from collections import Counter
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

start_time = time.time()
df = pd.read_csv("IMDB Dataset.csv").head(20000)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    words = [word for word in text.split() if word not in stop_words]
    return words

all_words = []
for review in df['review']:
    all_words.extend(clean_text(str(review)))

word_counts = Counter(all_words)
top_20 = word_counts.most_common(20)

pd.DataFrame(top_20, columns=['word', 'frequency']).to_csv('seq_output.csv', index=False)

print(f"Processed {len(df)} reviews in {time.time() - start_time:.2f} seconds.")
print("Top words:")
for word, freq in top_20:
    print(f"{word} ({freq})")
