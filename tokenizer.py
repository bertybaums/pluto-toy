"""
Word-level tokenizer for the toy corpus.

The corpus generator already separates punctuation with whitespace
("P10 is a planet ."), so a whitespace split gives us a clean ~60-token
vocabulary. That keeps the embedding matrix from dominating parameter
count in a Numerals-scale model.

Vocabulary is fit on the combined phase 1 + phase 2 corpus so that
entity names introduced only in phase 2 (E1..E5) already have IDs
during phase 1 training. Their embeddings are randomly initialized
and only get gradient updates once they appear in data — but they
exist throughout, which matters for the embedding-geometry probe.
"""
import json
from pathlib import Path


class WordTokenizer:
    def __init__(self, vocab=None):
        self.token_to_id = vocab or {}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    @classmethod
    def fit(cls, text):
        # token 0 reserved for <pad> so it shows up obviously in debugging
        vocab = {"<pad>": 0}
        for t in sorted(set(text.split())):
            vocab[t] = len(vocab)
        return cls(vocab)

    def encode(self, text):
        return [self.token_to_id[t] for t in text.split()]

    def decode(self, ids):
        return " ".join(self.id_to_token[i] for i in ids)

    def save(self, path):
        Path(path).write_text(json.dumps(self.token_to_id, indent=2))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    @property
    def vocab_size(self):
        return len(self.token_to_id)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True,
                    help="Text files to fit on (e.g., phase1.txt phase2.txt).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    text = "\n".join(Path(p).read_text() for p in args.corpus)
    tok = WordTokenizer.fit(text)
    tok.save(args.out)
    print(f"vocab_size: {tok.vocab_size}")
    print(f"sample tokens: {list(tok.token_to_id.items())[:15]}")


if __name__ == "__main__":
    main()
