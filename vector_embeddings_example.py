import os,requests
from importlib.metadata import version
import re

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

raw_text = None

## Download raw text

if not os.path.exists("the-verdict.txt"):
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    file_path = "the-verdict.txt"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)

# Reading raw text  
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

# Tokenizing text input. The regular expression will parse covering special characters including '--' which is commonly seen in texts
# Next we get the unique words that will form the vocabulary of the input data
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print("pre-processed text tokens: ", len(preprocessed))

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print("vocabulary size: ", vocab_size)
vocab = {token:integer for integer,token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
