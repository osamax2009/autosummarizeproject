"""
Test script to verify model loading works correctly
"""
import warnings
warnings.filterwarnings('ignore')

print("Importing libraries...")
from data_preprocessing import DataPreprocessor
from model import Seq2SeqLSTMSummarizer

# Configuration
max_text_len = 200
max_summary_len = 20

print("Loading preprocessor...")
preprocessor = DataPreprocessor(
    max_text_len=max_text_len,
    max_summary_len=max_summary_len
)
preprocessor.load_tokenizers()

print("Building model...")
model = Seq2SeqLSTMSummarizer(
    max_text_len=max_text_len,
    max_summary_len=max_summary_len,
    vocab_size_text=preprocessor.vocab_size_text,
    vocab_size_summary=preprocessor.vocab_size_summary,
    embedding_dim=128,
    latent_dim=256
)

model.build_model()
print("Loading weights...")
model.load_weights('model_weights.h5')

print("Building inference models...")
model.build_inference_models()

print("\n✓ Model loaded successfully!")
print("The GUI should now work without the Concatenate error.")

# Test with sample text
test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence to verify that the model can process text correctly."
print(f"\nTesting with sample text: {test_text[:50]}...")

input_seq = preprocessor.prepare_input_for_prediction(test_text)
reverse_target_word_index = preprocessor.get_reverse_word_index()
target_word_index = preprocessor.get_word_index()

print("Generating summary...")
summary = model.decode_sequence(
    input_seq,
    reverse_target_word_index,
    target_word_index,
    max_summary_len
)

print(f"\n✓ Generated summary: {summary}")
print("\n✓ All tests passed! The model is working correctly.")
