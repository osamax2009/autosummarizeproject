# Setup and Installation Guide

## Quick Start Guide for LSTM Text Summarization

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- At least 8GB RAM recommended
- GPU (optional, but recommended for faster training)

### Step-by-Step Installation

#### 1. Verify Python Installation

```bash
python --version
```

Should show Python 3.7 or higher.

#### 2. Create Virtual Environment (Recommended)

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (Deep learning framework)
- NumPy (Numerical computations)
- Pandas (Data manipulation)
- NLTK (Natural language processing)
- scikit-learn (Machine learning utilities)

#### 4. Download NLTK Data (Optional)

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### Training the Model

#### Option 1: Quick Training (Recommended for Testing)

For quick testing with a smaller dataset (10,000 samples):

```bash
python quick_train.py
```

This takes approximately 30-60 minutes on a modern CPU.

#### Option 2: Full Training

For better quality summaries using more data (50,000 samples):

```bash
python train.py
```

This takes several hours depending on your hardware.

#### Option 3: Custom Training

Customize training parameters:

```bash
python train.py --sample_size 20000 --epochs 10 --batch_size 32
```

**Training Parameters Explained:**

- `--sample_size`: Number of articles to train on
  - Smaller = faster training, lower quality
  - Larger = slower training, better quality
  - Recommended: 10,000 (testing), 50,000+ (production)

- `--epochs`: Number of training iterations
  - Default: 10
  - More epochs = better learning, but risk of overfitting

- `--batch_size`: Number of samples processed together
  - Larger = faster training, more memory usage
  - Smaller = slower training, less memory usage
  - Default: 64

- `--max_text_len`: Maximum length of input articles (in tokens)
  - Default: 200

- `--max_summary_len`: Maximum length of output summaries (in tokens)
  - Default: 20

### Using the GUI

After training, launch the GUI:

```bash
python gui.py
```

**GUI Features:**

1. **Input Text Area** (Left Panel)
   - Type or paste article text
   - Load from text file
   - Clear button

2. **Summary Output** (Right Panel)
   - View generated summary
   - Copy to clipboard
   - Save to file

3. **Generate Button**
   - Click to create summary
   - Shows progress indicator
   - Displays status messages

### Using the Command-Line Interface

For programmatic usage or testing:

```bash
python example_usage.py
```

This provides:
- Example demonstration
- Interactive mode for testing multiple texts

### File Structure After Setup

```
cnn_dailymail/
├── train.csv                  # Your training data
├── test.csv                   # Your test data
├── validation.csv             # Your validation data
├── model.py                   # Model architecture
├── data_preprocessing.py      # Data utilities
├── train.py                   # Training script
├── quick_train.py            # Quick training script
├── gui.py                     # GUI application
├── example_usage.py          # CLI example
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── SETUP_GUIDE.md           # This file
├── model_weights.h5         # Generated after training
├── x_tokenizer.pickle       # Generated after training
└── y_tokenizer.pickle       # Generated after training
```

### Troubleshooting

#### Problem: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution:**
```bash
pip install tensorflow
```

#### Problem: "Model weights not found"

**Solution:** Train the model first:
```bash
python quick_train.py
```

#### Problem: "Out of Memory" during training

**Solutions:**
1. Reduce batch size:
   ```bash
   python train.py --batch_size 32
   ```

2. Reduce sample size:
   ```bash
   python train.py --sample_size 5000
   ```

3. Close other applications to free up RAM

#### Problem: Training is very slow

**Solutions:**
1. Use smaller sample size for testing:
   ```bash
   python quick_train.py
   ```

2. Install TensorFlow GPU version (if you have NVIDIA GPU):
   ```bash
   pip install tensorflow-gpu
   ```

3. Reduce max_text_len:
   ```bash
   python train.py --max_text_len 100
   ```

#### Problem: Poor quality summaries

**Solutions:**
1. Train with more data:
   ```bash
   python train.py --sample_size 100000
   ```

2. Train for more epochs:
   ```bash
   python train.py --epochs 20
   ```

3. Ensure training completed successfully (check for model_weights.h5)

#### Problem: GUI window not appearing

**Solutions:**
1. Ensure tkinter is installed:
   ```bash
   python -m tkinter
   ```

2. On Linux, install tkinter:
   ```bash
   sudo apt-get install python3-tk
   ```

3. On macOS, reinstall Python with Homebrew:
   ```bash
   brew install python-tk
   ```

### Performance Tips

#### For Faster Training:
- Use `quick_train.py` for initial testing
- Start with small sample size (5,000-10,000)
- Use GPU if available
- Close unnecessary applications

#### For Better Quality:
- Use larger sample size (50,000+)
- Train for more epochs (15-20)
- Use full dataset if time permits
- Validate on test set

#### For Memory Efficiency:
- Reduce batch size (32 or 16)
- Reduce sample size
- Close other applications
- Use smaller vocabulary size

### Testing Your Installation

Run this quick test:

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

Expected output:
```
TensorFlow version: 2.x.x
GPU available: True/False
```

### Next Steps

1. **Train the model:**
   ```bash
   python quick_train.py
   ```

2. **Launch the GUI:**
   ```bash
   python gui.py
   ```

3. **Start summarizing!**
   - Enter or paste text
   - Click "Generate Summary"
   - View and save results

### Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Verify all dependencies are installed
3. Ensure dataset files (train.csv, validation.csv) exist
4. Check that you have enough disk space and RAM
5. Review error messages carefully

### Typical Training Times

**On Modern CPU (Intel i7/AMD Ryzen):**
- 5,000 samples: ~15 minutes
- 10,000 samples: ~30-60 minutes
- 50,000 samples: ~3-5 hours

**On GPU (NVIDIA GTX/RTX):**
- 5,000 samples: ~5 minutes
- 10,000 samples: ~10-15 minutes
- 50,000 samples: ~45-90 minutes

Times vary based on hardware specifications and other parameters.

### Recommended Workflow

#### For Learning/Testing:
1. Install dependencies
2. Run `quick_train.py` (10,000 samples)
3. Test with `gui.py`
4. Experiment with different texts

#### For Production Use:
1. Install dependencies
2. Run full training with 50,000+ samples
3. Validate results on test set
4. Deploy with `gui.py` or integrate into your application

### System Requirements

**Minimum:**
- CPU: Dual-core 2GHz+
- RAM: 8GB
- Storage: 5GB free space
- Python 3.7+

**Recommended:**
- CPU: Quad-core 3GHz+ or GPU
- RAM: 16GB+
- Storage: 10GB+ free space
- Python 3.8+
- CUDA-capable GPU (optional)

### Support

For issues specific to:
- **TensorFlow:** https://www.tensorflow.org/guide
- **Dataset:** Ensure CNN/DailyMail CSV format is correct
- **Python:** Verify Python version and virtual environment

Happy summarizing!
