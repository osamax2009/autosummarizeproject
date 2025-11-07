"""
QUICK MVP TRAINING - 15-20 minutes for demo purposes
This creates a working model fast, but quality will be lower
"""

import os
import sys
from data_preprocessing import prepare_training_data
from model import Seq2SeqLSTMSummarizer
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("QUICK MVP MODEL TRAINING (15-20 MINUTES)")
print("="*70)

print("\nThis will create a WORKING model in ~15-20 minutes for demo purposes.")
print("Quality will be lower than full training, but it will produce REAL")
print("summaries instead of <OOV> tokens.")
print("\nFor production, you should train with more data (6-12 hours).")

print("\n" + "="*70)
response = input("\nProceed with quick MVP training? (yes/no): ")
if response.lower() not in ['yes', 'y']:
    print("Cancelled.")
    sys.exit(0)

# Delete old mismatched files
print("\nCleaning up old files...")
for f in ['model_weights.h5', 'x_tokenizer.pickle', 'y_tokenizer.pickle']:
    if os.path.exists(f):
        os.remove(f)
        print(f"  ✓ Deleted {f}")

print("\n[Step 1/3] Loading data (small sample for speed)...")
# Use only 5,000 samples for quick training
preprocessor, training_data, validation_data = prepare_training_data(
    train_path='train.csv',
    val_path='validation.csv',
    sample_size=5000,  # Small for 15-20 min training
    max_text_len=200,
    max_summary_len=20
)

print(f"\n✓ Data ready:")
print(f"  - Training: {len(training_data['encoder_input'])} samples")
print(f"  - Validation: {len(validation_data['encoder_input'])} samples")
print(f"  - Text vocab: {preprocessor.vocab_size_text:,}")
print(f"  - Summary vocab: {preprocessor.vocab_size_summary:,}")

print("\n[Step 2/3] Building smaller model (for speed)...")
# Smaller model for faster training
model = Seq2SeqLSTMSummarizer(
    max_text_len=200,
    max_summary_len=20,
    vocab_size_text=preprocessor.vocab_size_text,
    vocab_size_summary=preprocessor.vocab_size_summary,
    embedding_dim=64,    # Reduced from 128
    latent_dim=128       # Reduced from 256
)

model.build_model()
model.compile_model(optimizer='adam', loss='sparse_categorical_crossentropy')

print(f"\n✓ Model built:")
print(f"  - Parameters: {model.model.count_params():,} (reduced for speed)")

print("\n[Step 3/3] Training (this will take ~15-20 minutes)...")
print("="*70)

# Prepare data
x_train = [training_data['encoder_input'], training_data['decoder_input']]
y_train = training_data['decoder_output']
x_val = [validation_data['encoder_input'], validation_data['decoder_input']]
y_val = validation_data['decoder_output']

# Train with fewer epochs and larger batch size for speed
import time
start_time = time.time()

history = model.train(
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    epochs=5,      # Just 5 epochs for speed
    batch_size=128,  # Larger batch = faster
    model_path='model_weights.h5'
)

elapsed = (time.time() - start_time) / 60

print("\n" + "="*70)
print("TRAINING COMPLETED!")
print("="*70)

print(f"\n✓ Training time: {elapsed:.1f} minutes")
print(f"✓ Final training loss: {history.history['loss'][-1]:.4f}")
print(f"✓ Final validation loss: {history.history['val_loss'][-1]:.4f}")

print("\n" + "="*70)
print("MVP MODEL READY!")
print("="*70)

print("\nNOTE: This is a quick demo model. Quality may be limited because:")
print("  - Only 5,000 training samples (vs 50,000+ for production)")
print("  - Only 5 epochs (vs 15-20 for production)")
print("  - Smaller model size (faster but less capacity)")

print("\nBut it WILL produce real summaries instead of <OOV> tokens!")

print("\n✓ To test the model right now:")
print("  python example_usage.py")

print("\n✓ To use the GUI:")
print("  python gui.py")

print("\n✓ For better quality later, run:")
print("  python fix_and_retrain.py")

print("\n" + "="*70)
