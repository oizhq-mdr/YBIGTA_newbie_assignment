from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    
    # Load poem sentiment dataset
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    
    # Extract verses from train, validation, and test splits
    for split in ['train', 'validation', 'test']:
        verses = dataset[split]['verse_text']
        corpus.extend(verses)
    
    return corpus