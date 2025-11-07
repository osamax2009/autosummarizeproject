"""
Quick training script with smaller dataset for testing
This script trains on a smaller sample for quick testing (10,000 samples)
"""

from train import train_model

if __name__ == '__main__':
    print("\n" + "="*70)
    print("QUICK TRAINING MODE - Testing with 10,000 samples")
    print("="*70)
    print("\nThis will train a model quickly for testing purposes.")
    print("For production use, run: python train.py --sample_size 50000 or more")
    print("\n" + "="*70 + "\n")

    # Train with smaller dataset for quick testing
    train_model(
        train_path='train.csv',
        val_path='validation.csv',
        sample_size=10000,  # Small sample for quick training
        max_text_len=200,
        max_summary_len=20,
        embedding_dim=128,
        latent_dim=256,
        epochs=5,  # Fewer epochs for quick training
        batch_size=64,
        model_save_path='model_weights.h5'
    )
