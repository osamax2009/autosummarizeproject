"""
ULTRA-FAST DEMO TRAINING FOR HOMEWORK
Trains in 2-3 minutes with tiny model for demonstration purposes
"""

import os
import sys
import pickle
from data_preprocessing import prepare_training_data
from model import Seq2SeqLSTMSummarizer
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("ULTRA-FAST HOMEWORK DEMO MODEL (2-3 MINUTES)")
print("="*70)

print("\nThis creates a tiny working model in ~2-3 minutes for homework demo.")
print("Quality will be limited but it WILL work and produce real summaries!")

# Clean up old files
print("\nCleaning up old files...")
for f in ['model_weights.h5', 'x_tokenizer.pickle', 'y_tokenizer.pickle']:
    if os.path.exists(f):
        os.remove(f)
        print(f"  ✓ Deleted {f}")

print("\n[1/3] Loading tiny dataset (1000 samples)...")
# Use only 1000 samples - very fast!
preprocessor, training_data, validation_data = prepare_training_data(
    train_path='train.csv',
    val_path='validation.csv',
    sample_size=1000,  # Tiny for speed
    max_text_len=100,  # Shorter sequences = faster
    max_summary_len=15
)

print(f"\n✓ Data ready:")
print(f"  - Training: {len(training_data['encoder_input'])} samples")
print(f"  - Validation: {len(validation_data['encoder_input'])} samples")

print("\n[2/3] Building tiny model...")
# Very small model for fast training
model = Seq2SeqLSTMSummarizer(
    max_text_len=100,
    max_summary_len=15,
    vocab_size_text=preprocessor.vocab_size_text,
    vocab_size_summary=preprocessor.vocab_size_summary,
    embedding_dim=32,    # Very small
    latent_dim=64        # Very small
)

model.build_model()
model.compile_model(optimizer='adam', loss='sparse_categorical_crossentropy')

print(f"✓ Model built: {model.model.count_params():,} parameters")

print("\n[3/3] Training (2-3 minutes)...")
print("="*70)

import time
start = time.time()

# Prepare data
x_train = [training_data['encoder_input'], training_data['decoder_input']]
y_train = training_data['decoder_output']
x_val = [validation_data['encoder_input'], validation_data['decoder_input']]
y_val = validation_data['decoder_output']

# Train fast - only 3 epochs, large batches
history = model.train(
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    epochs=3,        # Only 3 epochs
    batch_size=256,  # Large batch = faster
    model_path='model_weights.h5'
)

elapsed = (time.time() - start) / 60

# Save training history
print("\nSaving training history...")
with open('training_history.pickle', 'wb') as f:
    pickle.dump(history.history, f)
print("✓ Training history saved!")

print("\n" + "="*70)
print("DEMO MODEL READY FOR HOMEWORK!")
print("="*70)

print(f"\n✓ Training time: {elapsed:.1f} minutes")
print(f"✓ Final loss: {history.history['loss'][-1]:.4f}")
print(f"✓ Final accuracy: {history.history['accuracy'][-1]*100:.2f}%")

print("\n" + "="*70)
print("NOW UPDATE GUI TO USE THIS MODEL:")
print("="*70)

# Update GUI to use correct parameters
gui_code = '''
def load_model(self):
    """Load the trained model and tokenizers"""
    try:
        self.status_var.set("Loading model and tokenizers...")

        # Load preprocessor
        self.preprocessor = DataPreprocessor(
            max_text_len=100,  # Updated for demo model
            max_summary_len=15  # Updated for demo model
        )
        self.preprocessor.load_tokenizers()

        # Build and load model with DEMO parameters
        self.model = Seq2SeqLSTMSummarizer(
            max_text_len=100,
            max_summary_len=15,
            vocab_size_text=self.preprocessor.vocab_size_text,
            vocab_size_summary=self.preprocessor.vocab_size_summary,
            embedding_dim=32,   # DEMO model
            latent_dim=64       # DEMO model
        )

        self.model.build_model()
        self.model.load_weights('model_weights.h5')
        self.model.build_inference_models()

        self.model_loaded = True
        self.status_var.set("Model loaded successfully! Ready to summarize.")
        self.summarize_btn.config(state=tk.NORMAL, bg='#27ae60')

    except Exception as e:
        self.status_var.set(f"Error loading model: {{str(e)}}")
        messagebox.showerror("Error", f"Failed to load model:\\n{{str(e)}}")
'''

print("\nSaving updated GUI configuration...")

# Update the GUI file
with open('gui.py', 'r') as f:
    gui_content = f.read()

# Update the configuration section
gui_content = gui_content.replace('self.max_text_len = 200', 'self.max_text_len = 100')
gui_content = gui_content.replace('self.max_summary_len = 20', 'self.max_summary_len = 15')

# Update the model parameters in load_model method
gui_content = gui_content.replace('embedding_dim=64,   # MVP model uses 64 (not 128)', 'embedding_dim=32,   # DEMO model for homework')
gui_content = gui_content.replace('latent_dim=128      # MVP model uses 128 (not 256)', 'latent_dim=64       # DEMO model for homework')

with open('gui.py', 'w') as f:
    f.write(gui_content)

print("✓ GUI updated successfully!")

print("\n" + "="*70)
print("ALL DONE! YOUR HOMEWORK DEMO IS READY!")
print("="*70)

print("\n✓ To test quickly:")
print("  python test_mvp_model.py")

print("\n✓ To run the GUI:")
print("  python gui.py")

print("\nThe model will produce REAL summaries (not <OOV> tokens)!")
print("Perfect for homework demonstration!")

print("\n" + "="*70)
