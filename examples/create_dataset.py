# Create a proper test dataset
import os
from datasets import Dataset
from PIL import Image
import numpy as np

# Create test data directory if it doesn't exist
os.makedirs("./examples/test_data/train", exist_ok=True)
os.makedirs("./examples/test_data/test", exist_ok=True)

def create_sample_image(size=(64, 64)):
    """Create a simple sample image."""
    image_array = np.random.randint(0, 255, size + (3,), dtype=np.uint8)
    return Image.fromarray(image_array)

# Create training data
train_data = {
    "query": [
        "What is the main topic?",
        "How many sections are there?",
        "What is the conclusion?",
        "Who is the author?",
        "When was it published?"
    ],
    "image": [create_sample_image() for _ in range(5)]
}

train_dataset = Dataset.from_dict(train_data)
train_dataset.save_to_disk("./examples/test_data/train")

# Create test data
test_data = {
    "query": [
        "What are the key findings?",
        "What methodology was used?"
    ],
    "image": [create_sample_image() for _ in range(2)]
}

test_dataset = Dataset.from_dict(test_data)
test_dataset.save_to_disk("./examples/test_data/test")