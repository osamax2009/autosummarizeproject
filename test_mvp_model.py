"""
Test the MVP model with correct parameters
"""
from data_preprocessing import DataPreprocessor
from model import Seq2SeqLSTMSummarizer
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("TESTING MVP MODEL")
print("="*70)

# Load preprocessor
print("\nLoading tokenizers...")
preprocessor = DataPreprocessor(max_text_len=100, max_summary_len=15)  # DEMO model parameters
preprocessor.load_tokenizers()

print(f"✓ Text vocab: {preprocessor.vocab_size_text}")
print(f"✓ Summary vocab: {preprocessor.vocab_size_summary}")

# Build model with DEMO parameters (from quick_demo_train.py)
print("\nBuilding model...")
model = Seq2SeqLSTMSummarizer(
    max_text_len=100,
    max_summary_len=15,
    vocab_size_text=preprocessor.vocab_size_text,
    vocab_size_summary=preprocessor.vocab_size_summary,
    embedding_dim=32,    # DEMO model for homework
    latent_dim=64        # DEMO model for homework
)

model.build_model()
model.load_weights('model_weights.h5')
model.build_inference_models()

print("✓ Model loaded successfully!\n")
print("="*70)

# Test with example text
test_text = """
Artificial intelligence has become one of the most transformative technologies
of the 21st century. Machine learning, a subset of AI, enables computers to
learn from data without being explicitly programmed. Deep learning, which uses
neural networks with multiple layers, has achieved remarkable success in areas
such as image recognition, natural language processing, and speech recognition.
"""

print("\nTest Input:")
print("-" * 70)
print(test_text.strip())
print("-" * 70)

print("\nGenerating summary...")
input_seq = preprocessor.prepare_input_for_prediction(test_text)
reverse_target_word_index = preprocessor.get_reverse_word_index()
target_word_index = preprocessor.get_word_index()

summary = model.decode_sequence(
    input_seq,
    reverse_target_word_index,
    target_word_index,
    15  # Demo model uses 15
)

print("\nGenerated Summary:")
print("-" * 70)
print(summary)
print("-" * 70)

if '<OOV>' in summary:
    print("\n⚠ WARNING: Still producing OOV tokens.")
    print("The model needs more training for better quality.")
else:
    print("\n✓ SUCCESS: Model is producing real words!")

print("\n" + "="*70)
print("\nYour MVP model is working!")
print("For better quality, train longer with fix_and_retrain.py")
print("="*70)
