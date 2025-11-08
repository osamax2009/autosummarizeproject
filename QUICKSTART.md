# Quick Start Guide - Enhanced Text Summarization GUI

## 🚀 Quick Launch (If Model Already Trained)

```bash
# Activate virtual environment
source venv/bin/activate

# Launch the GUI
python gui.py
```

The GUI will open with two tabs:
1. **📝 Summarization** - Generate summaries
2. **📊 Training Metrics** - View training performance

## 📚 Try It in 3 Steps

1. **Click an Example Button**: Choose "Technology News", "Sports Update", or "Weather Report"
2. **Click "Generate Summary"**: Wait a few seconds
3. **See the Result**: The summary appears in the right panel

## 🔧 If Model Not Found

If you see "Model not found" error, train the model first:

```bash
# Activate virtual environment
source venv/bin/activate

# Quick training (2-3 minutes)
python quick_demo_train.py
```

This will create:
- `model_weights.h5` - Trained model
- `x_tokenizer.pickle` - Text tokenizer
- `y_tokenizer.pickle` - Summary tokenizer
- `training_history.pickle` - Training metrics

## 📊 View Training Performance

After training, switch to the **"Training Metrics"** tab to see:

✅ **Model Information**
- Architecture details
- Vocabulary sizes
- Number of epochs

✅ **Loss Chart**
- Training loss (blue line)
- Validation loss (red line)
- Shows model learning progress

✅ **Accuracy Chart**
- Training accuracy (blue line)
- Validation accuracy (red line)
- Shows prediction accuracy improvement

✅ **Performance Summary**
- Final training/validation metrics
- Best performing epoch
- Accuracy percentages

## 💡 Tips

### For Best Results:
- Use complete sentences (not fragments)
- Text should be 50-200 words for optimal results
- The model works best with news-style text

### GUI Features:
- **Load from File**: Import text from .txt files
- **Copy Summary**: Copy result to clipboard
- **Save to File**: Export summary to .txt file
- **Clear**: Reset input area

## ⚙️ Model Configuration

Current model settings:
- Embedding Dimension: **32**
- Latent Dimension: **64**
- Max Input Length: **100 tokens**
- Max Summary Length: **15 tokens**

These are optimized for quick demo training. For production use, increase these values and train longer.

## 🐛 Troubleshooting

### "Module not found: matplotlib"
```bash
source venv/bin/activate
pip install matplotlib
```

### "Shape mismatch error"
This is now **FIXED**. The GUI uses the correct parameters (embedding_dim=32, latent_dim=64).

### "Model not loaded"
Run the training script first:
```bash
python quick_demo_train.py
```

### GUI doesn't open
Make sure you're using the virtual environment:
```bash
source venv/bin/activate
python gui.py
```

## 📝 Example Workflow

### Complete Demo Workflow:

1. **Train the model** (one-time, 2-3 minutes):
   ```bash
   source venv/bin/activate
   python quick_demo_train.py
   ```

2. **Launch GUI**:
   ```bash
   python gui.py
   ```

3. **Try example text**:
   - Click "Technology News" button
   - Click "Generate Summary"
   - See the AI-generated summary!

4. **Check training metrics**:
   - Click "Training Metrics" tab
   - View loss and accuracy charts
   - See how well the model learned

5. **Try your own text**:
   - Go back to "Summarization" tab
   - Clear the input
   - Paste your own article
   - Generate summary

## 🎯 What to Expect

### Summary Quality:
- This is a **demo model** trained on limited data
- Summaries will be coherent but may not be perfect
- Quality improves with more training data and epochs

### Training Metrics:
- Loss should **decrease** over epochs (good!)
- Accuracy should **increase** over epochs (good!)
- Validation metrics should track training metrics
- Some divergence is normal (overfitting)

## 🔄 Re-training

To train a new model:

1. **Delete old model** (optional):
   ```bash
   rm model_weights.h5 training_history.pickle
   ```

2. **Run training again**:
   ```bash
   python quick_demo_train.py
   ```

3. **Restart GUI** to load new model

## ✨ Key Features

### Summarization Tab:
- ✅ 3 pre-loaded example texts
- ✅ Load custom text from file
- ✅ Real-time summary generation
- ✅ Copy/save functionality
- ✅ Progress indicator

### Training Metrics Tab:
- ✅ Loss visualization
- ✅ Accuracy visualization
- ✅ Model architecture info
- ✅ Performance statistics
- ✅ Best epoch identification

## 📧 Questions?

If you encounter issues:
1. Check that virtual environment is activated
2. Verify matplotlib is installed
3. Ensure model is trained (check for .h5 file)
4. Review error messages in terminal

---

**Enjoy your enhanced text summarization GUI!** 🎉
