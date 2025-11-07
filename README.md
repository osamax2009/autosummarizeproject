# LSTM Text Summarization - CNN/DailyMail Dataset

An automatic text summarization system using LSTM (Long Short-Term Memory networks) with encoder-decoder architecture and attention mechanism, trained on the CNN/DailyMail dataset.

## Features

- **Sequence-to-Sequence LSTM Model** with Bidirectional LSTM layers
- **Attention Mechanism** for better context understanding
- **Pre-trained on CNN/DailyMail Dataset** containing news articles and summaries
- **User-friendly GUI** built with Tkinter
- **Real-time summarization** with progress indication
- **File import/export** capabilities

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

## Usage

### Step 1: Train the Model

Train the model using your CNN/DailyMail dataset:

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

## GUI Features

The GUI application provides:

1. **Input Panel:**
   - Large text area for entering articles
   - Load text from file button
   - Clear input button

2. **Output Panel:**
   - Display generated summaries
   - Copy to clipboard button
   - Save to file button

3. **Controls:**
   - Generate Summary button
   - Real-time status updates
   - Progress indicator during generation

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

**Important:** All three files are required for the GUI to work!

## Troubleshooting

**Issue: "Model not found" error**
- Solution: Train the model first using `python train.py`

**Issue: Out of memory during training**
- Solution: Reduce `--sample_size` or `--batch_size`

**Issue: Training is too slow**
- Solution: Use smaller sample size or enable GPU acceleration

**Issue: Poor quality summaries**
- Solution: Train for more epochs or use larger dataset

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
- Tkinter (usually included with Python)

## License

This project uses the CNN/DailyMail dataset. Please ensure you comply with the dataset's license terms.

## Credits

Built with TensorFlow and Keras for deep learning, using the encoder-decoder architecture with attention mechanism for abstractive text summarization.
