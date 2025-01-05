# Fine-Tuning-LLAMA-3.2

Fine-tuning the LLAMA 3.2 model on Custom Dataset.

## Dataset Details
- Source: Mental health sentiment classification dataset
- Format: CSV with statements and status labels
- Pre-processing: Word count analysis and balanced sampling
- Max samples per category: 5,000

## Project Overview
This project FINE-TUNE LLAMA 3.2 to classify mental health-related text into categories:
- Normal
- Depression
- Suicidal
- Anxiety
- Bipolar
- Stress
- Personality Disorder

## Model Performance
- Overall Accuracy: 84.89%

## Fine-tuning Configuration
```yaml
# Key Parameters
model: LLAMA3_2
lora_rank: 64
lora_alpha: 128
lora_attn_modules: ['q_proj', 'v_proj', 'output_proj']
apply_lora_to_mlp: True
apply_lora_to_output: False
lora_dropout: 0.0
batch_size: 4
learning_rate: 3e-4
epochs: 1

```

## Training Process
1. Dataset Preparation
   - Load and clean mental health dataset
   - Balance categories
   - Split into train/test sets

2. Model Fine-tuning
   - LoRA adaptation
   - Cosine learning rate scheduling
   - Gradient accumulation

3. Evaluation
   - Accuracy metrics
   - Classification report
   - Error analysis


## Requirements
```
torch
transformers
torchtune
torchao
huggingface_hub
pandas
numpy
tqdm
```

## Model Access
The model uploaded on Hugging Face Hub:

## Future Improvements
- Experiment with larger model variants
- Increase training epochs
- Add multi-label classification
- Implement confidence scores
- Expand dataset variety
