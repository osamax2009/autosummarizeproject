"""
Training script for LSTM-based text summarization model
"""

import os
import argparse
from data_preprocessing import prepare_training_data
from model import Seq2SeqLSTMSummarizer
import warnings
warnings.filterwarnings('ignore')


def train_model(train_path='train.csv',
                val_path='validation.csv',
                sample_size=50000,
                max_text_len=200,
                max_summary_len=20,
                embedding_dim=128,
                latent_dim=256,
                epochs=10,
                batch_size=64,
                model_save_path='model_weights.h5'):
    """
    Train the summarization model

    Args:
        train_path: Path to training CSV file
        val_path: Path to validation CSV file
        sample_size: Number of samples to use for training (reduce for faster training)
        max_text_len: Maximum length of input text
        max_summary_len: Maximum length of summary
        embedding_dim: Dimension of embedding layer
        latent_dim: Dimension of LSTM hidden state
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_save_path: Path to save model weights
    """

    print("=" * 70)
    print("LSTM TEXT SUMMARIZATION - TRAINING")
    print("=" * 70)

    # Step 1: Prepare data
    print("\n[Step 1/3] Preparing training data...")
    preprocessor, training_data, validation_data = prepare_training_data(
        train_path=train_path,
        val_path=val_path,
        sample_size=sample_size,
        max_text_len=max_text_len,
        max_summary_len=max_summary_len
    )

    # Step 2: Build model
    print("\n[Step 2/3] Building model...")
    model = Seq2SeqLSTMSummarizer(
        max_text_len=max_text_len,
        max_summary_len=max_summary_len,
        vocab_size_text=preprocessor.vocab_size_text,
        vocab_size_summary=preprocessor.vocab_size_summary,
        embedding_dim=embedding_dim,
        latent_dim=latent_dim
    )

    model.build_model()
    model.compile_model()

    print(f"\nModel built successfully!")
    print(f"Total parameters: {model.model.count_params():,}")

    # Step 3: Train model
    print("\n[Step 3/3] Training model...")
    print(f"Training samples: {len(training_data['encoder_input'])}")
    print(f"Validation samples: {len(validation_data['encoder_input'])}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print("\nTraining started... (This may take a while)")
    print("-" * 70)

    history = model.train(
        x_train=[training_data['encoder_input'], training_data['decoder_input']],
        y_train=training_data['decoder_output'],
        x_val=[validation_data['encoder_input'], validation_data['decoder_input']],
        y_val=validation_data['decoder_output'],
        epochs=epochs,
        batch_size=batch_size,
        model_path=model_save_path
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\nModel weights saved to: {model_save_path}")
    print(f"Tokenizers saved to: x_tokenizer.pickle and y_tokenizer.pickle")
    print("\nYou can now use the GUI to generate summaries!")
    print("Run: python gui.py")
    print("=" * 70)

    return model, preprocessor, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM Text Summarization Model')

    parser.add_argument('--train', type=str, default='train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val', type=str, default='validation.csv',
                        help='Path to validation CSV file')
    parser.add_argument('--sample_size', type=int, default=50000,
                        help='Number of samples to use (use smaller for faster training)')
    parser.add_argument('--max_text_len', type=int, default=200,
                        help='Maximum length of input text')
    parser.add_argument('--max_summary_len', type=int, default=20,
                        help='Maximum length of summary')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='LSTM latent dimension')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--model_path', type=str, default='model_weights.h5',
                        help='Path to save model weights')

    args = parser.parse_args()

    # Train the model
    train_model(
        train_path=args.train,
        val_path=args.val,
        sample_size=args.sample_size,
        max_text_len=args.max_text_len,
        max_summary_len=args.max_summary_len,
        embedding_dim=args.embedding_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_path
    )
