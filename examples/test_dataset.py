import os
from datasets import Dataset, DatasetDict
from PIL import Image
import numpy as np


def create_test_image(size=(32, 32)):
    """Create a simple test image."""
    # Create a simple red square image
    image_array = np.random.randint(0, 255, size + (3,), dtype=np.uint8)
    image = Image.fromarray(image_array)
    return image


def create_test_dataset(save_path="./test_data"):
    """Create a minimal test dataset for ColPali training verification."""
    
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Create sample data
    queries = [
        "What is the total revenue?",
        "How many employees are there?",
        "What is the company name?",
        "When was the report published?",
        "What is the net profit?"
    ]
    
    # Create sample images
    images = [create_test_image() for _ in range(len(queries))]
    
    # Create dataset dict
    dataset_dict = DatasetDict({
        "train": Dataset.from_dict({
            "query": queries,
            "image": images
        }),
        "test": Dataset.from_dict({
            "query": queries[:2],
            "image": images[:2]
        })
    })
    
    # Save dataset
    dataset_dict.save_to_disk(save_path)
    print(f"Test dataset saved to {save_path}")
    
    return dataset_dict


def create_test_dataset_for_detailed_training(save_path="./test_data_detailed"):
    """Create a test dataset compatible with the detailed training format."""
    
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Sample data for different domains
    datasets_info = {
        "infovqa_train": {
            "queries": [
                "What is written in the form?",
                "Who is the applicant?",
                "What is the date?"
            ]
        },
        "docvqa_train": {
            "queries": [
                "What is the contract value?",
                "Who are the parties involved?",
                "When does the contract expire?"
            ]
        },
        "arxivqa_train": {
            "queries": [
                "What is the main contribution?",
                "Which conference was it published in?",
                "What is the proposed method?"
            ]
        }
    }
    
    # Create dataset dicts for each domain
    all_datasets = {}
    for dataset_name, info in datasets_info.items():
        queries = info["queries"]
        images = [create_test_image((64, 64)) for _ in range(len(queries))]
        
        # For detailed training, we need train/test splits
        if len(queries) >= 2:
            train_dataset = Dataset.from_dict({
                "query": queries[1:],
                "image": images[1:]
            })
            
            test_dataset = Dataset.from_dict({
                "query": queries[:1],
                "image": images[:1]
            })
            
            dataset_dict = DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })
        else:
            # If not enough data, just duplicate
            dataset_dict = DatasetDict({
                "train": Dataset.from_dict({
                    "query": queries,
                    "image": images
                }),
                "test": Dataset.from_dict({
                    "query": queries,
                    "image": images
                })
            })
        
        full_path = os.path.join(save_path, dataset_name)
        os.makedirs(full_path, exist_ok=True)
        dataset_dict.save_to_disk(full_path)
        all_datasets[dataset_name] = dataset_dict
        print(f"Saved {dataset_name} to {full_path}")
    
    return all_datasets


if __name__ == "__main__":
    # Create basic test dataset
    create_test_dataset()
    
    # Create detailed training dataset
    create_test_dataset_for_detailed_training()
    
    print("Test datasets created successfully!")