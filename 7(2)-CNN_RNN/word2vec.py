import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal
from tqdm import tqdm # type: ignore

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!
        # pass

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        
        print("Tokenizing corpus...")
        tokenized_corpus = []
        for text in tqdm(corpus):
            tokens = tokenizer(text, add_special_tokens=False)['input_ids']
            if tokens:  # Only add if not empty
                tokenized_corpus.append(tokens)
        print(f"Tokenized {len(tokenized_corpus)} sentences")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            total_loss: float = 0.0 
            
            if self.method == "cbow":
                print("Training CBOW...")
                total_loss = self._train_cbow(tokenized_corpus, criterion, optimizer)
            else:  # skipgram
                print("Training Skip-gram...")
                total_loss = self._train_skipgram(tokenized_corpus, criterion, optimizer)
                
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    def _train_cbow(
        self,
        tokenized_corpus: list[list[int]],
        criterion: nn.Module,
        optimizer: Adam
    ) -> float:
        total_loss = 0
        
        for sentence in tqdm(tokenized_corpus):
            for i in range(len(sentence)):
                # Get context words
                context_words = []
                for j in range(-self.window_size, self.window_size + 1):
                    if j == 0:  # Skip target word
                        continue
                    idx = i + j
                    if 0 <= idx < len(sentence):
                        context_words.append(sentence[idx])
                
                if not context_words:  # Skip if no context words
                    continue
                
                # Convert to tensor
                context_tensor = torch.tensor(context_words, device=self.embeddings.weight.device)
                target_tensor = torch.tensor([sentence[i]], device=self.embeddings.weight.device)
                
                # Forward pass
                context_embeds = self.embeddings(context_tensor)
                context_mean = torch.mean(context_embeds, dim=0, keepdim=True)
                output = self.weight(context_mean)
                
                # Backward pass
                loss = criterion(output, target_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
        return total_loss

    def _train_skipgram(
        self,
        tokenized_corpus: list[list[int]],
        criterion: nn.Module,
        optimizer: Adam
    ) -> float:
        total_loss = 0
        
        for sentence in tqdm(tokenized_corpus):
            for i in range(len(sentence)):
                target_tensor = torch.tensor([sentence[i]], device=self.embeddings.weight.device)
                
                # Get context words
                for j in range(-self.window_size, self.window_size + 1):
                    if j == 0:  # Skip target word
                        continue
                    
                    idx = i + j
                    if 0 <= idx < len(sentence):
                        context_word = sentence[idx]
                        context_tensor = torch.tensor([context_word], device=self.embeddings.weight.device)
                        
                        # Forward pass
                        target_embed = self.embeddings(target_tensor)
                        output = self.weight(target_embed)
                        
                        # Backward pass
                        loss = criterion(output, context_tensor)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
        
        return total_loss

    # 구현하세요!
    pass