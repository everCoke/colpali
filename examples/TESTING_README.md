jiaoben# Testing ColPali Training Pipeline

This directory contains scripts and datasets for testing the ColPali training pipeline.

## Overview

We've created several test datasets and scripts to verify that the ColPali training pipeline works correctly:

1. **Basic test dataset**: A minimal dataset for quick testing
2. **Detailed test datasets**: Multiple domain-specific datasets similar to the actual training data
3. **Test scripts**: Various Python scripts to verify different aspects of the training pipeline

## Test Datasets

### Basic Test Dataset

Located in `./test_data`, this dataset contains:
- 5 training samples with queries and synthetic images
- 2 evaluation samples with queries and synthetic images

### Detailed Test Datasets

Located in `./test_data_detailed`, this collection contains domain-specific datasets:
- `infovqa_train`: Simulated InfoVQA-like data
- `docvqa_train`: Simulated DocVQA-like data  
- `arxivqa_train`: Simulated ArXivQA-like data

Each dataset includes both training and testing splits with synthetic queries and images.

## Test Scripts

### 1. `test_dataset.py`

Creates the test datasets described above.

Usage:
```bash
python test_dataset.py
```

### 2. `test_training.py`

Verifies basic components of the training pipeline without actually training a model.

Usage:
```bash
python test_training.py
```

### 3. `final_test.py`

Fully tests the training pipeline initialization with mocked components.

Usage:
```bash
python final_test.py
```

## How to Use for Training Verification

1. **Generate test data**:
   ```bash
   python test_dataset.py
   ```

2. **Verify pipeline components**:
   ```bash
   python final_test.py
   ```

3. **Check that all components work together**:
   The test scripts verify that:
   - Dataset loading works correctly
   - Data collation functions properly
   - Trainer initialization succeeds
   - Loss functions can be computed
   - Mock training loop executes

## Customization

You can customize the test datasets by modifying `test_dataset.py`:
- Change the number of samples
- Modify queries
- Adjust image sizes or patterns
- Add new domain-specific datasets

The test scripts are designed to be lightweight and fast, allowing you to quickly verify that changes to the training pipeline don't break core functionality.

## About Actual Training and Model Outputs

The test scripts use mocked components to verify the training pipeline without performing actual model training. In a real training scenario:

### Real Training Process

1. **Load pretrained models**:
   ```python
   processor = ColPaliProcessor.from_pretrained("google/paligemma-3b-mix-448")
   model = ColPali.from_pretrained("google/paligemma-3b-mix-448")
   ```

2. **Prepare real datasets** with queries and document images

3. **Configure training arguments**:
   - Batch size
   - Learning rate
   - Number of epochs
   - Evaluation schedule

4. **Initialize trainer** with real components

5. **Run training** which can take hours to days depending on dataset size and hardware

### Trained Model Structure

After successful training, the output directory will contain:
- Model weights (`pytorch_model.bin`)
- Configuration files (`config.json`)
- Processor files (`preprocessor_config.json`)
- Training logs and metrics
- Checkpoint files at various intervals

### Using Trained Models

Once trained, you can load and use the model:

```python
# Load the trained model
processor = ColPaliProcessor.from_pretrained("./path/to/trained/model")
model = ColPali.from_pretrained("./path/to/trained/model")

# Process queries and documents
query = "What is the total revenue?"
image = Image.open("document.png")

# Encode query and image
query_embedding = model.encode_queries(processor.process_queries([query]))
image_embedding = model.encode_images(processor.process_images([image]))

# Compute similarity
similarity = (query_embedding @ image_embedding.T)
```

### Hardware Requirements for Real Training

Real training requires significant computational resources:
- High-end GPUs (preferably with 24GB+ VRAM)
- Large amounts of system RAM (32GB+)
- Significant storage space for datasets and model checkpoints
- Potentially distributed training setup for large-scale training