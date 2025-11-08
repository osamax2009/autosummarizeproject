# LSTM Text Summarization - CNN/DailyMail Dataset

An automatic text summarization system using LSTM (Long Short-Term Memory networks) with encoder-decoder architecture and attention mechanism, trained on the CNN/DailyMail dataset.

## ✨ New Features (Enhanced Version)

- **📊 Training Metrics Visualization** - Interactive charts showing loss and accuracy
- **📝 Example Text Library** - Pre-loaded examples for instant testing
- **🎨 Tabbed Interface** - Separate tabs for summarization and metrics
- **📈 Performance Analytics** - View model training performance
- **🚀 Quick Launch Script** - One-command GUI startup

## Features

- **Sequence-to-Sequence LSTM Model** with Bidirectional LSTM layers
- **Attention Mechanism** for better context understanding
- **Pre-trained on CNN/DailyMail Dataset** containing news articles and summaries
- **Enhanced GUI** with training metrics visualization
- **Real-time summarization** with progress indication
- **File import/export** capabilities
- **Training history tracking** with matplotlib charts

## Project Structure

```
cnn_dailymail/
├── train.csv              # Training dataset
├── test.csv               # Test dataset
├── validation.csv         # Validation dataset
├── model.py               # LSTM model architecture
├── data_preprocessing.py  # Data preprocessing utilities
├── train.py               # Training script
├── gui.py                 # GUI application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download NLTK data (optional, for stopwords removal):**
```python
python -c "import nltk; nltk.download('stopwords')"
```

## Quick Start

### Fast Demo (Recommended for Testing)

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Quick training (2-3 minutes)
python quick_demo_train.py

# 3. Launch enhanced GUI
./run_gui.sh
# Or: python gui.py
```

See **[QUICKSTART.md](QUICKSTART.md)** for detailed instructions.

## Usage

### Step 1: Train the Model

**Option A: Quick Demo Training (2-3 minutes)**
```bash
python quick_demo_train.py
```

**Option B: Full Training (longer, better quality)**
```bash
python train.py
```

**Training Options:**
```bash
python train.py --sample_size 50000 --epochs 10 --batch_size 64
```

**Available arguments:**
- `--train`: Path to training CSV (default: train.csv)
- `--val`: Path to validation CSV (default: validation.csv)
- `--sample_size`: Number of samples to use (default: 50000)
- `--max_text_len`: Maximum text length (default: 200)
- `--max_summary_len`: Maximum summary length (default: 20)
- `--embedding_dim`: Embedding dimension (default: 128)
- `--latent_dim`: LSTM hidden dimension (default: 256)
- `--epochs`: Number of epochs (default: 10)
- `--batch_size`: Batch size (default: 64)
- `--model_path`: Path to save weights (default: model_weights.h5)

**Note:** Training on the full dataset takes considerable time. Start with `--sample_size 10000` for quick testing.

### Step 2: Run the GUI

After training, launch the GUI application:

```bash
python gui.py
```

## Enhanced GUI Features

The enhanced GUI application provides two tabs:

### Tab 1: 📝 Summarization
1. **Example Text Buttons:**
   - Technology News
   - Sports Update
   - Weather Report
   - One-click loading for instant testing

2. **Input Panel:**
   - Large text area for entering articles
   - Load text from file button
   - Clear input button

3. **Output Panel:**
   - Display generated summaries
   - Copy to clipboard button
   - Save to file button

4. **Controls:**
   - Generate Summary button
   - Real-time status updates
   - Progress indicator during generation

### Tab 2: 📊 Training Metrics
1. **Model Information:**
   - Architecture details
   - Vocabulary sizes
   - Training configuration

2. **Training Charts:**
   - Loss over epochs (training vs validation)
   - Accuracy over epochs (training vs validation)
   - Interactive matplotlib plots

3. **Performance Summary:**
   - Final training/validation metrics
   - Best epoch identification
   - Accuracy percentages

## Model Architecture

The model uses a **Sequence-to-Sequence architecture** with:

- **Encoder:**
  - Embedding layer (128 dimensions)
  - 2 Bidirectional LSTM layers (256 units each)
  - Dropout for regularization (0.4)

- **Decoder:**
  - Embedding layer (128 dimensions)
  - LSTM layer (512 units)
  - Attention mechanism
  - Dense output layer with softmax

- **Training:**
  - Loss: Sparse Categorical Crossentropy
  - Optimizer: RMSprop
  - Early stopping with patience=3
  - Model checkpoint to save best weights

## Dataset Format

The CSV files should have the following columns:
- `id`: Unique identifier
- `article`: The full news article text
- `highlights`: The summary/highlights of the article

## Example Usage

1. **Train with custom parameters:**
```bash
python train.py --sample_size 20000 --epochs 15 --batch_size 32
```

2. **Use the trained model:**
```bash
python gui.py
```

3. **Enter or paste an article in the input area**

4. **Click "Generate Summary" to create a summary**

## Performance Tips

- **For faster training:** Reduce `--sample_size` (e.g., 10000)
- **For better quality:** Increase `--epochs` and use full dataset
- **Memory issues:** Reduce `--batch_size` or `--sample_size`
- **GPU acceleration:** Ensure TensorFlow GPU is properly installed

## Files Generated During Training

- `model_weights.h5`: Trained model weights
- `x_tokenizer.pickle`: Input text tokenizer
- `y_tokenizer.pickle`: Summary tokenizer
- `training_history.pickle`: **NEW** - Training metrics for visualization

**Important:** All files are required for full GUI functionality!

## Troubleshooting

**Issue: "Shape mismatch" error when loading model**
- Solution: ✅ **FIXED** - GUI now uses correct parameters (embedding_dim=32, latent_dim=64)

**Issue: "Model not found" error**
- Solution: Train the model first using `python quick_demo_train.py`

**Issue: "matplotlib not found" error**
- Solution: `source venv/bin/activate && pip install matplotlib`

**Issue: Out of memory during training**
- Solution: Reduce `--sample_size` or `--batch_size`

**Issue: Training is too slow**
- Solution: Use `quick_demo_train.py` for fast 2-3 minute training

**Issue: Poor quality summaries**
- Solution: Train for more epochs or use larger dataset

**Issue: No training metrics showing**
- Solution: Delete old model files and retrain to generate `training_history.pickle`

## Technical Details

- **Input:** Articles up to 200 tokens
- **Output:** Summaries up to 20 tokens
- **Vocabulary:** Top 10,000 most frequent words
- **Special tokens:** `sostok` (start of summary), `eostok` (end of summary)

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- Pandas
- NumPy
- NLTK
- Matplotlib (for training charts)
- Tkinter (usually included with Python)

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide and common workflows
- **[CHANGES.md](CHANGES.md)** - Detailed list of all enhancements and fixes
- **[run_gui.sh](run_gui.sh)** - Convenient launcher script

## License

This project uses the CNN/DailyMail dataset. Please ensure you comply with the dataset's license terms.

## Credits

Built with TensorFlow and Keras for deep learning, using the encoder-decoder architecture with attention mechanism for abstractive text summarization.
