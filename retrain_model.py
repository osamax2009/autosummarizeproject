"""
Retrain the model with better parameters to fix OOV output issue
"""

import os
import sys
from data_preprocessing import prepare_training_data
from model import Seq2SeqLSTMSummarizer
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("RETRAINING MODEL TO FIX OOV OUTPUT ISSUE")
print("="*70)

print("\nIMPORTANT:")
print("- This will take SEVERAL HOURS (possibly 8-15 hours)")
print("- Do NOT interrupt the training")
print("- Your computer may slow down during training")
print("- You should see the loss decreasing over epochs")
print("\n" + "="*70)

response = input("\nDo you want to proceed with retraining? (yes/no): ")
if response.lower() not in ['yes', 'y']:
    print("Training cancelled.")
    sys.exit(0)

print("\n[Step 1/3] Preparing training data...")
print("Using 100,000 samples for better learning...")

preprocessor, training_data, validation_data = prepare_training_data(
    train_path='train.csv',
    val_path='validation.csv',
    sample_size=100000,  # Increased from default 50000
    max_text_len=200,
    max_summary_len=20
)

print(f"\n✓ Training data prepared:")
print(f"  - Training samples: {len(training_data['encoder_input'])}")
print(f"  - Validation samples: {len(validation_data['encoder_input'])}")
print(f"  - Text vocabulary size: {preprocessor.vocab_size_text}")
print(f"  - Summary vocabulary size: {preprocessor.vocab_size_summary}")

print("\n[Step 2/3] Building model...")
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

print("✓ Model built and compiled")
print(f"  - Total parameters: {model.model.count_params():,}")

print("\n[Step 3/3] Training model...")
print("This will take SEVERAL HOURS. Progress will be shown below.")
print("="*70)

# Prepare data for training (model expects specific format)
x_train = [training_data['encoder_input'], training_data['decoder_input']]
y_train = training_data['decoder_output']
x_val = [validation_data['encoder_input'], validation_data['decoder_input']]
y_val = validation_data['decoder_output']

# Train with more epochs (20 instead of 10)
history = model.train(
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    epochs=20,  # Increased from 10
    batch_size=64,
    model_path='model_weights.h5'
)

print("\n" + "="*70)
print("TRAINING COMPLETED!")
print("="*70)

print("\nFinal training metrics:")
print(f"  - Final training loss: {history.history['loss'][-1]:.4f}")
print(f"  - Final validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"  - Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  - Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

print("\nModel saved to: model_weights.h5")
print("\n✓ You can now use the GUI - it should produce real summaries!")
print("\nTo test the model, run:")
print("  python example_usage.py")
print("\nTo start the GUI, run:")
print("  python gui.py")
print("\n" + "="*70)
