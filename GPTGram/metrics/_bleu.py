from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from ._base import BaseMetric

class BLEU(BaseMetric):
    def __init__(self, n_gram=4):
        self.weights = [1.0 / n_gram] * n_gram
    
    def score(self, references, candidate):
        """
        Calculates the BLEU score for the given reference and candidate sentences.
        
        Args:
            references (list of list of str): The list of reference sentences. Each sentence is a list of words.
            candidate (list of str): The candidate sentence, which is a list of words.
            
        Returns:
            float: The calculated BLEU score.
        """
        # Convert the candidate and references to the expected format if necessary
        if isinstance(candidate[0], str):
            candidate = [candidate]
        references = [[ref.split()] for ref in references]
        candidate = [candidate[0].split()]

        # Calculate and return the BLEU score
        return corpus_bleu(references, candidate, weights=self.weights)