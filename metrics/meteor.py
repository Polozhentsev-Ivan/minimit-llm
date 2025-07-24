import nltk
from typing import List
from nltk.translate.meteor_score import single_meteor_score

class MeteorMetric:
    def __init__(self):
        for pkg in ('wordnet', 'omw-1.4'):
            try:
                nltk.data.find(pkg)
            except LookupError:
                nltk.download(pkg, quiet=True)

    def _score(self, ref: str, hyp: str) -> float:
        ref_tokens = str.split(ref)
        hyp_tokens = str.split(hyp)
        return single_meteor_score(ref_tokens, hyp_tokens)

    def compute(self, predictions: List[str], references: List[str]) -> float:
        scores = [self._score(r, p) for p, r in zip(predictions, references)]
        return sum(scores) / len(scores) if scores else 0.0

    def compute_scores(self, predictions: List[str], references: List[str]) -> List[float]:
        return [self._score(r, p) for p, r in zip(predictions, references)]