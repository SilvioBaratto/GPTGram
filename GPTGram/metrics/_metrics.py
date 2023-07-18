import nltk
from rouge import Rouge
import torch
from torch.nn import functional as F

class Metrics:
    """
    You can create an instance of the Metrics class and call the evaluate_metrics method by passing the predicted and actual outputs:

    Example:
        metrics = Metrics(model, data_loader)  # Pass the trained model and data loader
        metrics_dict = metrics.evaluate_metrics(predicted_output, actual_output)

        # Access the individual metric values:
        f1_score = metrics_dict["f1_score"]
        bleu_score = metrics_dict["bleu_score"]
        rouge_1 = metrics_dict["rouge_1"]
        rouge_2 = metrics_dict["rouge_2"]
        rouge_l = metrics_dict["rouge_l"]

        print(f"F1 Score: {f1_score:.2f}")
        print(f"BLEU Score: {bleu_score:.2f}")
        print(f"ROUGE-1: {rouge_1:.2f}")
        print(f"ROUGE-2: {rouge_2:.2f}")
        print(f"ROUGE-L: {rouge_l:.2f}")
        print(f"Perplexity: {perplexity:.2f}")
"""
    def __init__(self, model=None, data_loader=None):
        self.model = model
        self.data_loader = data_loader

    def evaluate_metrics(self, predicted_output=None, actual_output=None):
        metrics = {}

        if predicted_output is not None and actual_output is not None:
            f1_score = self.calculate_f1(predicted_output, actual_output)
            bleu_score = self.calculate_bleu(predicted_output, actual_output)
            rouge_1, rouge_2, rouge_l = self.calculate_rouge(predicted_output, actual_output)
            metrics["f1_score"] = f1_score
            metrics["bleu_score"] = bleu_score
            metrics["rouge_1"] = rouge_1
            metrics["rouge_2"] = rouge_2
            metrics["rouge_l"] = rouge_l
        
        if self.data_loader is not None:
            perplexity = self.calculate_perplexity()
            metrics["perplexity"] = perplexity

        return metrics

    def calculate_f1(self, predicted, actual):
        predicted_tokens = predicted.split()
        actual_tokens = actual.split()

        common_tokens = set(predicted_tokens) & set(actual_tokens)
        precision = len(common_tokens) / len(predicted_tokens) if len(predicted_tokens) > 0 else 0
        recall = len(common_tokens) / len(actual_tokens) if len(actual_tokens) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1

    def calculate_bleu(self, predicted, actual):
        # Convert the input sentences into lists of tokens
        predicted_tokens = predicted.split()
        actual_tokens = actual.split()

        # Calculate BLEU score for 4-grams (BLEU-4)
        weights = (0.25, 0.25, 0.25, 0.25)  # Equal weights for 1-gram, 2-gram, 3-gram, and 4-gram
        bleu_score = nltk.translate.bleu_score.sentence_bleu([actual_tokens], predicted_tokens, weights)
        return bleu_score

    def calculate_rouge(self, predicted, actual):
        rouge = Rouge()

        # Calculate ROUGE score (ROUGE-1, ROUGE-2, and ROUGE-L)
        scores = rouge.get_scores(predicted, actual)
        rouge_1 = scores[0]['rouge-1']['f']
        rouge_2 = scores[0]['rouge-2']['f']
        rouge_l = scores[0]['rouge-l']['f']

        return rouge_1, rouge_2, rouge_l

    def calculate_perplexity(self):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.data_loader:
                inputs = batch["input"]  # Assuming the input data is stored in an "input" key
                targets = batch["target"]  # Assuming the target data is stored in a "target" key
                inputs = inputs.to(self.model.device)
                targets = targets.to(self.model.device)

                logits, _ = self.model(inputs, targets)  # Forward pass to get logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                total_loss += loss.item() * targets.numel()  # Multiply the loss by the number of tokens
                total_tokens += targets.numel()  # Count the number of tokens

        average_loss = total_loss / total_tokens
        perplexity = 2 ** average_loss

        return perplexity


