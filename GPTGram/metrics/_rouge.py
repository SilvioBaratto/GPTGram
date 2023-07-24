from ._base import BaseMetric
from rouge_score import rouge_scorer

class Rouge(BaseMetric):
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def score(self, references, candidate):
        """
        Calculates the ROUGE-L score for the given reference and candidate sentences.

        Args:
            references (list of str): The list of reference sentences.
            candidate (str): The candidate sentence.

        Returns:
            float: The calculated ROUGE-L score.
        """
        scores = [self.scorer.score(reference, candidate)['rougeL'].fmeasure for reference in references]
        return max(scores) if scores else 0