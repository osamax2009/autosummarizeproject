"""
Example usage of the LSTM Text Summarization model
This script demonstrates how to use the trained model programmatically
"""

from data_preprocessing import DataPreprocessor
from model import Seq2SeqLSTMSummarizer
import os


def summarize_text(text, model, preprocessor, max_summary_len=20):
    """
    Summarize a given text using the trained model

    Args:
        text: Input text to summarize
        model: Trained Seq2SeqLSTMSummarizer model
        preprocessor: DataPreprocessor instance with loaded tokenizers
        max_summary_len: Maximum length of summary

    Returns:
        Generated summary as string
    """
    # Prepare input
    input_seq = preprocessor.prepare_input_for_prediction(text)

    # Get word indices
    reverse_target_word_index = preprocessor.get_reverse_word_index()
    target_word_index = preprocessor.get_word_index()

    # Generate summary
    summary = model.decode_sequence(
        input_seq,
        reverse_target_word_index,
        target_word_index,
        max_summary_len
    )

    return summary


def main():
    """Main function to demonstrate usage"""

    # Check if model files exist
    if not os.path.exists('model_weights.h5'):
        print("Error: Model weights not found!")
        print("Please train the model first by running: python train.py")
        return

    print("\n" + "="*70)
    print("LSTM TEXT SUMMARIZATION - EXAMPLE USAGE")
    print("="*70)

    # Load preprocessor
    print("\nLoading preprocessor and tokenizers...")
    preprocessor = DataPreprocessor(max_text_len=200, max_summary_len=20)
    preprocessor.load_tokenizers()

    # Build and load model
    print("Loading model...")
    model = Seq2SeqLSTMSummarizer(
        max_text_len=200,
        max_summary_len=20,
        vocab_size_text=preprocessor.vocab_size_text,
        vocab_size_summary=preprocessor.vocab_size_summary,
        embedding_dim=128,
        latent_dim=256
    )

    model.build_model()
    model.load_weights('model_weights.h5')
    model.build_inference_models()

    print("Model loaded successfully!\n")
    print("="*70)

    # Example text to summarize
    example_text = """
    Artificial intelligence has become one of the most transformative technologies
    of the 21st century. Machine learning, a subset of AI, enables computers to
    learn from data without being explicitly programmed. Deep learning, which uses
    neural networks with multiple layers, has achieved remarkable success in areas
    such as image recognition, natural language processing, and speech recognition.
    Companies across various industries are implementing AI solutions to improve
    efficiency, reduce costs, and enhance customer experiences. However, the rapid
    advancement of AI also raises important ethical questions about privacy, job
    displacement, and algorithmic bias that society must address.
    """

    print("\nOriginal Text:")
    print("-" * 70)
    print(example_text.strip())
    print("-" * 70)

    print("\nGenerating summary...")
    summary = summarize_text(example_text, model, preprocessor)

    print("\nGenerated Summary:")
    print("-" * 70)
    print(summary)
    print("-" * 70)

    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter your text to summarize (or 'quit' to exit)")
    print("-" * 70)

    while True:
        print("\nYour text (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            else:
                if lines:
                    break

        user_text = ' '.join(lines)

        if user_text.lower() == 'quit':
            print("\nGoodbye!")
            break

        if user_text.strip():
            print("\nGenerating summary...")
            summary = summarize_text(user_text, model, preprocessor)
            print("\nSummary:")
            print("-" * 70)
            print(summary)
            print("-" * 70)
        else:
            print("No text entered. Type 'quit' to exit or enter text to summarize.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nMake sure you have trained the model first:")
        print("python train.py")
