from torchvision import datasets

from datasets import load_dataset

# Load the ArXiv dataset (you can specify the field/topic if necessary)
dataset = load_dataset("arxiv", split="train[:10%]")  # For example, using a subset
