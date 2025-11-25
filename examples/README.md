# ColPali Training Examples

This directory contains example scripts and test datasets for verifying the ColPali training pipeline.

## Directory Structure

```
examples/
├── test_dataset.py              # Script to generate test datasets
├── test_training.py             # Basic training pipeline verification
├── final_test.py                # Complete training pipeline test with mock components
├── real_training_example.py     # Example of how to set up real model training
├── test_data/                   # Basic test dataset with queries and synthetic images
├── test_data_detailed/          # Detailed domain-specific test datasets
└── README.md                    # This file
```

## Test Datasets

### Basic Test Dataset (`test_data/`)

A minimal dataset containing:
- 5 training samples with queries and synthetic images
- 2 evaluation samples with queries and synthetic images

### Detailed Test Datasets (`test_data_detailed/`)

Domain-specific datasets similar to actual training data:
- `infovqa_train/`: Simulated InfoVQA-like data
- `docvqa_train/`: Simulated DocVQA-like data  
- `arxivqa_train/`: Simulated ArXivQA-like data

Each dataset includes both training and testing splits with synthetic queries and images.

## Example Scripts

### 1. Test Dataset Generation (`test_dataset.py`)

Creates the test datasets described above.

Usage:
```bash
python test_dataset.py
```

### 2. Basic Training Verification (`test_training.py`)

Verifies basic components of the training pipeline without actually training a model.

Usage:
```bash
python test_training.py
```

### 3. Complete Pipeline Test (`final_test.py`)

Fully tests the training pipeline initialization with mocked components.

Usage:
```bash
python final_test.py
```

### 4. Real Training Example (`real_training_example.py`)

Demonstrates how to set up real model training with actual ColPali models.

Usage:
```bash
python real_training_example.py
```

## Usage

To test the training pipeline:

1. **Generate test data**:
   ```bash
   python test_dataset.py
   ```

2. **Verify pipeline components**:
   ```bash
   python final_test.py
   ```

These tests verify that:
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

## About Actual Training

The test scripts use mocked components to verify the training pipeline without performing actual model training. For real training, see `real_training_example.py` for guidance on setting up training with actual ColPali models.

Real training requires:
- Powerful GPU(s) with significant VRAM
- Large amounts of training data
- Proper dataset setup
- Potentially distributed training setup for large-scale training