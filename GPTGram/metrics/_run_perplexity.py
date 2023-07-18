from ..preprocessing import GramDataset
import os
from torch.utils.data import DataLoader
from . import Metrics

# Output file paths
train_output_file = os.path.join('test.bin')

# Create a dataset instance
dataset = GramDataset(train_output_file)

# Crate dataloader
dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle = False)
metrics = Metrics(model, dataloader)  # Pass the trained model and data loader
metrics_dict = metrics.evaluate_metrics()
perplexity = metrics_dict["perplexity"]
print(f"Perplexity: {perplexity:.2f}")