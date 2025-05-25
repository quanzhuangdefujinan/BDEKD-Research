# BDEKD: Backdoor Defense via Ensemble Knowledge Distillation

This repository contains the main experiment code for the paper "BDEKD: Mitigating Backdoor Attacks in NLP Models via Ensemble Knowledge Distillation." The code implements the BDEKD framework for backdoor purification in NLP models, including teacher diversification, data augmentation, and ensemble knowledge distillation.

## Overview

The BDEKD framework mitigates backdoor attacks in NLP models by:
1. **Teacher Diversification**: Generating three heterogeneous teacher models using fine-tuning, re-initialization, and fine-pruning.
2. **Data Augmentation**: Expanding clean data via synonym replacement and random word insertion, preserving key semantic words.
3. **Ensemble Distillation**: Distilling knowledge from multiple teachers to a student model, erasing backdoors while preserving clean accuracy.

This repository provides the core implementation (`main.py`), excluding backdoor models and datasets to focus on the purification methodology.

## Repository Structure

- `main.py`: Main script implementing the BDEKD pipeline.
- `requirements.txt`: Python dependencies required to run the code.
- `README.md`: This file, providing setup and usage instructions.

## Prerequisites

- Python 3.8+
- CUDA-enabled GPU (optional, for faster training)
- Clean dataset in text format (e.g., SST-2 format: text and binary labels)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BDEKD-Research/BDEKD.git
   cd BDEKD
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download NLTK WordNet data:
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

## Usage

1. **Prepare Clean Data**:
   - Provide a clean dataset with text samples and labels (e.g., CSV with columns `text` and `label`).
   - Example format:
     ```python
     texts = ["A sad, superior human comedy played out on the back roads of life.", 
              "This movie is absolutely fantastic and inspiring."]
     labels = [0, 1]  # 0: Negative, 1: Positive
     ```

2. **Run the BDEKD Pipeline**:
   - Modify `main.py` to load your dataset (replace placeholder `texts` and `labels` in the `__main__` block).
   - Execute the script:
     ```bash
     python main.py
     ```
   - The script will:
     - Augment the data using synonym replacement and random insertion.
     - Generate three teacher models via fine-tuning, re-initialization, and fine-pruning.
     - Perform ensemble distillation to produce a purified student model.
     - Output the trained student model.

3. **Evaluate the Model**:
   - The script outputs a purified student model (`BertForSequenceClassification`).
   - Users can evaluate the model on their own test set (not provided) to measure Attack Success Rate (ASR) and Clean Accuracy (CA).

## Notes

- **Excluded Components**: Backdoor models and datasets (e.g., SST-2, HS, OLID2, AG News) are not included, as they are not essential for reproducing the BDEKD purification process. Users must provide their own clean data and, if desired, poisoned models for evaluation.
- **Model Architecture**: The code uses BERT-base-uncased by default, consistent with the paper’s victim model (Section 4.1.2). Modify `model_name` in `main.py` for other architectures (e.g., BERT-large), ensuring structural similarity (Section 4.3.2).
- **Hyperparameters**:
  - Fine-tuning: Learning rate 0.1, halved every 2 epochs (Appendix B, Page 33).
  - Re-initialization: Layers 7–11 for BERT-base (Appendix B).
  - Fine-pruning: Threshold 0.7 (Appendix C, Page 34).
  - Distillation: Alpha=0.5, beta=0.5, temperature=2.0 (Section 3.3.3).
- **Dependencies**: Listed in `requirements.txt`. Ensure a CUDA-enabled GPU for optimal performance, though CPU is supported.
- **Limitations**: The code assumes a binary classification task (e.g., SST-2). For multi-class tasks (e.g., AG News), adjust the `num_labels` parameter in model initialization.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{BDEKD2025,
  title={BDEKD: Mitigating Backdoor Attacks in NLP Models via Ensemble Knowledge Distillation},
  author={Author Names},
  journal={Nuclear Physics B},
  year={2025}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact [your_email@example.com].
