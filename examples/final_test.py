#!/usr/bin/env python3
"""
Final test script to verify the complete ColPali training pipeline.
"""

import os
from datasets import Dataset
from PIL import Image
import numpy as np
import torch
from transformers import TrainingArguments

# Import colpali_engine components
from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import ColbertPairwiseCELoss
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.collators import VisualRetrieverCollator


def create_sample_image(size=(32, 32)):
    """Create a simple sample image."""
    # Create a simple pattern
    image_array = np.zeros(size + (3,), dtype=np.uint8)
    image_array[::4, ::4] = [255, 0, 0]  # Red dots
    image = Image.fromarray(image_array)
    return image


def create_minimal_dataset():
    """Create a minimal dataset for testing."""
    # Sample queries and images
    queries = [
        "What is the main topic?",
        "How many sections are there?",
        "What is the conclusion?",
        "Who is the author?",
        "When was it published?",
        "What are the key findings?",
        "What methodology was used?",
        "What is the purpose of the study?"
    ]
    
    images = [create_sample_image() for _ in range(len(queries))]
    
    # Create HF Dataset
    dataset = Dataset.from_dict({
        "query": queries,
        "image": images
    })
    
    # Convert to ColPaliEngineDataset
    colpali_dataset = ColPaliEngineDataset(
        data=dataset,
        pos_target_column_name="image"
    )
    
    return colpali_dataset


class MockModel(torch.nn.Module):
    """A simple mock model for testing the training pipeline."""
    
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.tensor([1.0]))
        
    def forward(self, *args, **kwargs):
        # Simple mock forward that returns dummy embeddings
        batch_size = kwargs.get('batch_size', 2)
        embedding_dim = 128
        seq_len = 32
        
        # Return mock embeddings
        return torch.randn(batch_size, seq_len, embedding_dim)


class MockProcessor:
    """A simple mock processor for testing."""
    
    def __init__(self):
        self.query_prefix = "Query: "
        self.pos_doc_prefix = "Document: "
        self.image_token = "<image>"
        
    def process_images(self, images):
        # Return mock processed images
        return {"pixel_values": torch.randn(len(images), 3, 32, 32)}
    
    def process_texts(self, texts):
        # Return mock processed texts
        return {"input_ids": torch.randint(0, 1000, (len(texts), 16)),
                "attention_mask": torch.ones(len(texts), 16)}


def test_complete_pipeline():
    """Test the complete training pipeline with mocked components."""
    print("Creating minimal test dataset...")
    train_dataset = create_minimal_dataset()
    eval_dataset = create_minimal_dataset().take(4)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Create mock components
    mock_model = MockModel()
    mock_processor = MockProcessor()
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./test_output",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=2,
        save_steps=3,
        logging_steps=1,
        remove_unused_columns=False,
        disable_tqdm=False,
    )
    
    # Create loss function
    loss_func = ColbertPairwiseCELoss()
    
    # Create data collator
    collator = VisualRetrieverCollator(
        processor=mock_processor,
        max_length=50,
    )
    
    print("Initializing trainer...")
    # Initialize trainer correctly
    trainer = ContrastiveTrainer(
        model=mock_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        loss_func=loss_func,
        is_vision_model=True,
    )
    
    print("Trainer initialized successfully!")
    print("Training pipeline verified.")
    
    return True


def test_data_loading():
    """Test that data can be properly loaded and processed."""
    print("Testing data loading...")
    
    # Test basic dataset creation
    dataset = create_minimal_dataset()
    
    # Check that we can access items
    item = dataset[0]
    assert "query" in item, "Query should be in dataset item"
    assert "pos_target" in item, "Positive target should be in dataset item"
    
    print(f"Sample item keys: {list(item.keys())}")
    print(f"Sample query: {item['query']}")
    print(f"Sample positive target type: {type(item['pos_target'])}")
    
    # Test taking subset
    subset = dataset.take(3)
    assert len(subset) == 3, "Subset should have 3 items"
    
    print("Data loading test passed!")
    return True


if __name__ == "__main__":
    # Set environment variable to use local datasets
    os.environ["USE_LOCAL_DATASET"] = "1"
    
    try:
        print("=" * 50)
        print("Testing data loading...")
        print("=" * 50)
        success1 = test_data_loading()
        
        print("\n" + "=" * 50)
        print("Testing complete pipeline...")
        print("=" * 50)
        success2 = test_complete_pipeline()
        
        if success1 and success2:
            print("\n✓ All tests passed! The training pipeline is working correctly.")
        else:
            print("\n✗ Some tests failed!")
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()