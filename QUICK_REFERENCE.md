# Quick Reference Card

## Essential Commands

### First Time Setup
```bash
# 1. Check if everything is ready
python3 check_setup.py

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (choose one)
python3 quick_train.py          # Fast: 10K samples, ~30-60 min
python3 train.py                # Standard: 50K samples, ~3-5 hours
```

### Daily Usage
```bash
# Launch GUI
python3 gui.py

# Use command line
python3 example_usage.py
```

## File Reference

### Run These Files
| File | When to Use |
|------|-------------|
| `check_setup.py` | Verify installation |
| `quick_train.py` | Train quickly (testing) |
| `train.py` | Train properly (production) |
| `gui.py` | Use GUI interface |
| `example_usage.py` | Use command line |

### Read These Files
| File | Information |
|------|-------------|
| `README.md` | Project overview |
| `SETUP_GUIDE.md` | Detailed setup instructions |
| `PROJECT_SUMMARY.md` | Complete project details |
| `QUICK_REFERENCE.md` | This file |

### Don't Run These (They're Imported)
- `model.py`
- `data_preprocessing.py`

## Training Options

### Quick Training (Testing)
```bash
python3 quick_train.py
# 10,000 samples, 5 epochs, ~30-60 minutes
```

### Custom Training
```bash
python3 train.py \
  --sample_size 50000 \
  --epochs 10 \
  --batch_size 64 \
  --max_text_len 200 \
  --max_summary_len 20
```

## Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sample_size` | 50000 | Number of training samples |
| `--epochs` | 10 | Training iterations |
| `--batch_size` | 64 | Samples per batch |
| `--max_text_len` | 200 | Max input length |
| `--max_summary_len` | 20 | Max summary length |

## Troubleshooting

### Problem: Dependencies not installed
```bash
pip install -r requirements.txt
```

### Problem: Model not found
```bash
python3 quick_train.py  # Train the model first
```

### Problem: Out of memory
```bash
python3 train.py --batch_size 32 --sample_size 10000
```

### Problem: Training too slow
```bash
python3 quick_train.py  # Use smaller dataset
```

### Problem: GUI won't start (macOS)
```bash
brew install python-tk
```

### Problem: GUI won't start (Linux)
```bash
sudo apt-get install python3-tk
```

## Expected Training Times

| Samples | CPU | GPU |
|---------|-----|-----|
| 5,000 | 15 min | 5 min |
| 10,000 | 45 min | 15 min |
| 50,000 | 4 hrs | 90 min |

## Project Structure

```
cnn_dailymail/
├── train.csv                 # Dataset (don't modify)
├── validation.csv            # Dataset (don't modify)
├── test.csv                  # Dataset (don't modify)
│
├── model.py                  # Model code (don't run)
├── data_preprocessing.py     # Data code (don't run)
│
├── train.py                  # Run: Train model
├── quick_train.py           # Run: Quick train
├── gui.py                   # Run: GUI app
├── example_usage.py         # Run: CLI demo
├── check_setup.py           # Run: Verify setup
│
├── README.md                # Read: Overview
├── SETUP_GUIDE.md          # Read: Setup help
├── PROJECT_SUMMARY.md      # Read: Full details
├── QUICK_REFERENCE.md      # Read: This file
│
├── requirements.txt        # Dependencies
│
└── Generated after training:
    ├── model_weights.h5         # Model weights
    ├── x_tokenizer.pickle       # Input tokenizer
    └── y_tokenizer.pickle       # Output tokenizer
```

## Workflow Summary

```
1. python3 check_setup.py         ← Verify environment
2. pip install -r requirements.txt ← Install packages
3. python3 quick_train.py         ← Train model
4. python3 gui.py                 ← Use the app!
```

## GUI Features

- **Left Panel**: Input text area
- **Right Panel**: Generated summary
- **Bottom Button**: "Generate Summary"
- **File Operations**: Load/Save
- **Clipboard**: Copy summary

## Status Messages

| Message | Meaning |
|---------|---------|
| "Model loaded successfully" | Ready to use |
| "Model not found" | Need to train |
| "Generating summary..." | Working |
| "Summary generated successfully" | Done |

## System Requirements

**Minimum**: Python 3.7+, 8GB RAM, 5GB disk
**Recommended**: Python 3.8+, 16GB RAM, GPU

## Quality Tips

✓ More samples = Better quality
✓ More epochs = Better learning
✓ GPU = Faster training
✓ Test with quick_train.py first
✓ Use full training for production

## Help Resources

1. **Quick check**: `python3 check_setup.py`
2. **Setup help**: Read `SETUP_GUIDE.md`
3. **Full details**: Read `PROJECT_SUMMARY.md`
4. **Try example**: `python3 example_usage.py`

## One-Line Summary

**Purpose**: Automatically summarize text using LSTM trained on CNN/DailyMail news dataset

---

Keep this file handy for quick reference!
