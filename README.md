# SAM Fine-tuning with FFT Self-Attention on COCO Dataset

This repository implements fine-tuning of the Segment Anything Model (SAM) using COCO dataset with a novel FFT-based self-attention mechanism. The project replaces traditional self-attention layers in SAM with Fast Fourier Transform (FFT) based attention for improved computational efficiency.

## Features

- **FFT Self-Attention**: Custom implementation of self-attention using Fast Fourier Transform for efficient computation
- **SAM Model Modification**: Automated replacement of standard attention layers with FFT-based versions
- **COCO Dataset Integration**: Automatic download and preprocessing of COCO validation dataset
- **Cross-platform Support**: Compatible with CUDA, MPS (Apple Silicon), and CPU devices
- **Memory Optimized**: Configurable batch sizes and dataset subsets for different hardware capabilities

## Architecture Overview

### FFT Self-Attention Mechanism
The core innovation replaces traditional self-attention computation with FFT-based operations:
- Query and Key projections are transformed to frequency domain using `torch.fft.rfft`
- Complex multiplication in frequency domain (equivalent to convolution in time domain)
- Inverse FFT to return to spatial domain
- Combined with Value vectors for final attention output

### Modified SAM Architecture
- Base SAM model with ViT-B backbone
- FFT attention layers replace all MultiheadAttention modules
- Additional segmentation head for multi-class prediction
- Support for 91 classes (COCO dataset: 90 classes + background)

## Requirements

```bash
pip install torch torchvision
pip install segment-anything
pip install pycocotools
pip install Pillow
pip install tqdm
pip install numpy
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bemoregt/Finetuning_SAM_usingCOCO.git
cd Finetuning_SAM_usingCOCO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
Simply run the main script to start training:
```bash
python main.py
```

The script will automatically:
1. Download COCO validation dataset (images + annotations)
2. Download SAM ViT-B model checkpoint
3. Replace attention layers with FFT versions
4. Start fine-tuning process

### Configuration Options

#### Device Selection
The code automatically detects and uses the best available device:
- CUDA (if available)
- MPS (Apple Silicon)
- CPU (fallback)

#### Memory Management
For systems with limited memory, adjust these parameters:
```python
# Reduce batch size
batch_size = 1  # Default: 1

# Limit dataset size
subset_size = 100  # Default: 100 samples

# Reduce image resolution
target_size = (512, 512)  # Default: (1024, 1024)
```

#### Training Parameters
```python
num_epochs = 3      # Number of training epochs
learning_rate = 1e-5  # Adam optimizer learning rate
num_workers = 0      # DataLoader workers (0 for stability)
```

## Model Architecture Details

### FFTSelfAttention Class
```python
class FFTSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        # Query, Key, Value projection layers
        # FFT-based attention computation
        # Output projection
```

Key features:
- Multi-head attention support
- Configurable dropout
- Efficient FFT operations using `torch.fft`

### ModifiedSAM Class
```python
class ModifiedSAM(nn.Module):
    def __init__(self, sam_model, num_classes=91):
        # SAM image encoder
        # Custom segmentation head
        # Feature upsampling
```

Architecture flow:
1. Input image → SAM Image Encoder → Feature maps [B, 256, H/16, W/16]
2. Feature upsampling → Original resolution [B, 256, H, W]
3. Segmentation head → Class predictions [B, num_classes, H, W]

## Dataset

### COCO Dataset Structure
- **Images**: COCO 2017 validation set (~5K images)
- **Annotations**: Instance segmentation masks
- **Classes**: 90 object categories + background
- **Format**: JSON annotations with polygon segmentations

### Automatic Download
The script automatically downloads:
- `val2017.zip` (validation images, ~1GB)
- `annotations_trainval2017.zip` (annotations, ~241MB)

Downloaded files are cached locally to avoid re-downloading.

## Training Process

### Data Pipeline
1. **Image Loading**: PIL Image → RGB conversion
2. **Mask Generation**: COCO polygons → Binary masks → Multi-class masks
3. **Preprocessing**: Resize to target size + Normalization
4. **Batching**: Custom collate function for variable-sized inputs

### Loss Function
- **CrossEntropyLoss**: Multi-class segmentation loss
- **Optimization**: Adam optimizer with learning rate 1e-5

### Training Loop
```python
for epoch in range(num_epochs):
    for images, masks in dataloader:
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
```

## Performance Considerations

### Memory Usage
- **Batch Size**: Set to 1 to prevent OOM errors
- **Image Resolution**: 1024x1024 (adjustable)
- **Dataset Subset**: Limited to 100 samples by default

### Computational Efficiency
- **FFT Operations**: O(n log n) complexity vs O(n²) for standard attention
- **Device Optimization**: Automatic device selection
- **Gradient Checkpointing**: Can be added for further memory savings

## Output

### Model Checkpoint
Trained model is saved as:
```
fft_sam_finetuned.pth
```

### Training Metrics
Console output includes:
- Dataset preparation progress
- Model loading status
- Training loss per epoch
- Device information

## File Structure

```
Finetuning_SAM_usingCOCO/
├── main.py                 # Main training script
├── README.md              # This file
├── LICENSE                # License file
├── coco_data/            # Auto-created COCO dataset directory
│   ├── val2017/          # Validation images
│   └── annotations/      # COCO annotations
├── sam_checkpoints/      # Auto-created SAM model directory
│   └── sam_vit_b.pth    # SAM ViT-B checkpoint
└── fft_sam_finetuned.pth # Output trained model
```

## Technical Details

### FFT Self-Attention Mathematics
The FFT attention mechanism implements:
1. **Frequency Transform**: Q, K → Frequency domain
2. **Complex Multiplication**: Attention in frequency space
3. **Inverse Transform**: Back to spatial domain
4. **Value Integration**: Combine with value vectors

### Model Replacement Strategy
The `replace_attention_with_fft()` function:
- Recursively traverses SAM model
- Identifies `nn.MultiheadAttention` modules
- Replaces with `FFTSelfAttention` maintaining same parameters
- Preserves model architecture integrity

## Limitations

- **Memory Requirements**: Large images require significant GPU memory
- **Dataset Size**: Full COCO dataset may be too large for some systems
- **Training Time**: Fine-tuning SAM is computationally intensive
- **FFT Approximation**: FFT attention is an approximation of standard attention

## Future Improvements

- [ ] Add validation metrics (IoU, mAP)
- [ ] Implement mixed precision training
- [ ] Support for custom datasets
- [ ] Model quantization for deployment
- [ ] Distributed training support
- [ ] Tensorboard logging integration

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Meta AI**: Original SAM model implementation
- **COCO Dataset**: Microsoft Common Objects in Context
- **PyTorch**: Deep learning framework
- **Fast Fourier Transform**: Mathematical foundation for efficient attention

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{fft-sam-finetune,
  title={SAM Fine-tuning with FFT Self-Attention on COCO Dataset},
  author={bemoregt},
  year={2025},
  publisher={GitHub},
  url={https://github.com/bemoregt/Finetuning_SAM_usingCOCO}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.
