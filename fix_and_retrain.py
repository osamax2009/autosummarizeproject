"""
Clean up mismatched files and retrain everything from scratch
This fixes the vocabulary size mismatch issue
"""

import os
import sys
from data_preprocessing import prepare_training_data
from model import Seq2SeqLSTMSummarizer
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("FIX VOCABULARY MISMATCH AND RETRAIN MODEL")
print("="*70)

print("\nPROBLEM DETECTED:")
print("Your tokenizers and model weights don't match!")
print("  - Current tokenizers: vocab_text=280,680, vocab_summary=98,087")
print("  - Old model weights: vocab_text=97,229, vocab_summary=35,778")
print("\nSOLUTION:")
print("We need to delete the old incompatible files and retrain from scratch.")

print("\n" + "="*70)

# Check if old files exist
old_files = []
if os.path.exists('model_weights.h5'):
    old_files.append('model_weights.h5')
if os.path.exists('x_tokenizer.pickle'):
    old_files.append('x_tokenizer.pickle')
if os.path.exists('y_tokenizer.pickle'):
    old_files.append('y_tokenizer.pickle')

if old_files:
    print("\nThe following files will be DELETED and recreated:")
    for f in old_files:
        size = os.path.getsize(f) / (1024*1024)  # MB
        print(f"  - {f} ({size:.1f} MB)")

    print("\n" + "="*70)
    response = input("\nProceed with cleanup and retraining? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        sys.exit(0)

    # Delete old files
    print("\nDeleting old incompatible files...")
    for f in old_files:
        os.remove(f)
        print(f"  ✓ Deleted {f}")

print("\n" + "="*70)
print("STARTING FRESH TRAINING")
print("="*70)

print("\nNOTE:")
print("- Training will take 6-12 HOURS")
print("- Do NOT close this window or interrupt the process")
print("- You can monitor progress below")
print("\n" + "="*70)

print("\n[Step 1/3] Preparing training data from scratch...")
print("Loading 50,000 samples (reduced for faster training)...")

# Use 50,000 samples for reasonable training time
preprocessor, training_data, validation_data = prepare_training_data(
    train_path='train.csv',
    val_path='validation.csv',
    sample_size=50000,  # Reasonable size for 6-12 hour training
    max_text_len=200,
    max_summary_len=20
)

print(f"\n✓ Data prepared and tokenizers created:")
print(f"  - Training samples: {len(training_data['encoder_input'])}")
print(f"  - Validation samples: {len(validation_data['encoder_input'])}")
print(f"  - Text vocabulary: {preprocessor.vocab_size_text}")
print(f"  - Summary vocabulary: {preprocessor.vocab_size_summary}")

print("\n[Step 2/3] Building model with correct vocabulary sizes...")
model = Seq2SeqLSTMSummarizer(
    max_text_len=200,
    max_summary_len=20,
    vocab_size_text=preprocessor.vocab_size_text,
    vocab_size_summary=preprocessor.vocab_size_summary,
    embedding_dim=128,
    latent_dim=256
)

model.build_model()
model.compile_model(optimizer='adam', loss='sparse_categorical_crossentropy')

print(f"\n✓ Model built:")
print(f"  - Total parameters: {model.model.count_params():,}")

print("\n[Step 3/3] Training model...")
print("="*70)
print("TRAINING IN PROGRESS - This will take several hours")
print("="*70)

# Prepare data
x_train = [training_data['encoder_input'], training_data['decoder_input']]
y_train = training_data['decoder_output']
x_val = [validation_data['encoder_input'], validation_data['decoder_input']]
y_val = validation_data['decoder_output']

# Train for 15 epochs (good balance between time and quality)
history = model.train(
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    epochs=15,
    batch_size=64,
    model_path='model_weights.h5'
)

print("\n" + "="*70)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)

print("\nFinal results:")
print(f"  - Training loss: {history.history['loss'][-1]:.4f}")
print(f"  - Validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"  - Training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  - Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

print("\n✓ Files created:")
print("  - model_weights.h5 (trained model)")
print("  - x_tokenizer.pickle (text tokenizer)")
print("  - y_tokenizer.pickle (summary tokenizer)")

print("\n" + "="*70)
print("ALL DONE! Your model should now work correctly!")
print("="*70)

print("\nTo test the model:")
print("  python example_usage.py")
print("\nTo use the GUI:")
print("  python gui.py")

print("\nYou should now see REAL summaries instead of <OOV> tokens!")
print("\n" + "="*70)
