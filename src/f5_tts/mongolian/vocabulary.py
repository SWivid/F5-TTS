# vocabulary.py

from typing import List

class Vocabulary:
    def __init__(self, specials: List[str] = None):
        self.token2idx = {}
        self.idx2token = {}
        self.next_index = 0
        self.specials = specials or []
        self.embedding_dim = 512  # Match with your position embedding dimension

        # Add special tokens first
        for token in self.specials:
            self.add_token(token)
        # Add '<unk>' token for unknown tokens
        self.add_token('<unk>')
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

    def add_token(self, token: str):
        if token not in self.token2idx:
            self.token2idx[token] = self.next_index
            self.idx2token[self.next_index] = token
            self.next_index += 1
        return self.token2idx[token]

    def __getitem__(self, token: str):
        # Return the index of the token, or index of '<unk>' if token not found
        return self.token2idx.get(token, self.token2idx[self.unk_token])

    def __len__(self):
        return len(self.token2idx)
