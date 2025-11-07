"""
Debug script to inspect model layer structure
"""
import warnings
warnings.filterwarnings('ignore')

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

print("\nBuilding model...")
model = Seq2SeqLSTMSummarizer(
    max_text_len=max_text_len,
    max_summary_len=max_summary_len,
    vocab_size_text=preprocessor.vocab_size_text,
    vocab_size_summary=preprocessor.vocab_size_summary,
    embedding_dim=128,
    latent_dim=256
)

model.build_model()
print("\nLoading weights...")
model.load_weights('model_weights.h5')

print("\n" + "="*60)
print("MODEL LAYER STRUCTURE:")
print("="*60)
for i, layer in enumerate(model.model.layers):
    print(f"Layer {i}: {layer.name:30s} | {type(layer).__name__:20s} | {layer.output.shape if hasattr(layer.output, 'shape') else 'N/A'}")

print("\n" + "="*60)
print("Building inference models...")
print("="*60)

try:
    model.build_inference_models()
    print("\n✓ Inference models built successfully!")

    # Test with sample text
    test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence."
    print(f"\nTesting with: {test_text}")

    input_seq = preprocessor.prepare_input_for_prediction(test_text)
    reverse_target_word_index = preprocessor.get_reverse_word_index()
    target_word_index = preprocessor.get_word_index()

    print("\nGenerating summary...")
    summary = model.decode_sequence(
        input_seq,
        reverse_target_word_index,
        target_word_index,
        max_summary_len
    )

    print(f"\n✓ Generated summary: {summary}")

    if '<OOV>' in summary or not summary.strip():
        print("\n⚠ WARNING: Summary contains OOV tokens or is empty!")
        print("This indicates a weight transfer issue.")
    else:
        print("\n✓ SUCCESS: Summary looks good!")

except Exception as e:
    print(f"\n✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()
