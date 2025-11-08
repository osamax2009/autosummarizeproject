"""
Flask Web Application for LSTM Text Summarization
Modern HTML interface with training metrics visualization
"""

from flask import Flask, render_template, request, jsonify
import os
import pickle
import json
from data_preprocessing import DataPreprocessor
from model import Seq2SeqLSTMSummarizer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model
model = None
preprocessor = None
model_loaded = False
training_history = None

# Configuration
MAX_TEXT_LEN = 100
MAX_SUMMARY_LEN = 15

# Example texts
EXAMPLE_TEXTS = [
    {
        "id": "tech",
        "title": "Technology News",
        "text": "Apple announced its latest iPhone at a special event yesterday. The new device features improved cameras, faster processor, and longer battery life. The company CEO highlighted the enhanced AI capabilities and new design. Pre-orders start next week with delivery expected in two weeks. Analysts predict strong sales for the holiday season."
    },
    {
        "id": "sports",
        "title": "Sports Update",
        "text": "The championship game ended with a dramatic finish last night. The home team scored in the final seconds to win by three points. Fans celebrated in the streets after the historic victory. The star player was named MVP of the game. This marks the team's first championship in over twenty years."
    },
    {
        "id": "weather",
        "title": "Weather Report",
        "text": "A major storm system is approaching the coast bringing heavy rain and strong winds. Forecasters warn of potential flooding in low-lying areas. Residents are advised to prepare emergency supplies and stay indoors. The storm is expected to last through the weekend. Schools and businesses may close on Friday."
    }
]


def load_model():
    """Load the trained model and tokenizers"""
    global model, preprocessor, model_loaded

    try:
        # Load preprocessor
        preprocessor = DataPreprocessor(
            max_text_len=MAX_TEXT_LEN,
            max_summary_len=MAX_SUMMARY_LEN
        )
        preprocessor.load_tokenizers()

        # Build and load model
        model = Seq2SeqLSTMSummarizer(
            max_text_len=MAX_TEXT_LEN,
            max_summary_len=MAX_SUMMARY_LEN,
            vocab_size_text=preprocessor.vocab_size_text,
            vocab_size_summary=preprocessor.vocab_size_summary,
            embedding_dim=32,   # Match trained model
            latent_dim=64       # Match trained model
        )

        model.build_model()
        model.load_weights('model_weights.h5')
        model.build_inference_models()

        model_loaded = True
        print("✓ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model_loaded = False
        return False


def load_training_history():
    """Load training history from pickle file"""
    global training_history

    history_path = 'training_history.pickle'
    if os.path.exists(history_path):
        try:
            with open(history_path, 'rb') as f:
                training_history = pickle.load(f)
            print("✓ Training history loaded")
            return True
        except Exception as e:
            print(f"✗ Could not load training history: {e}")
            return False
    return False


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', examples=EXAMPLE_TEXTS)


@app.route('/api/status')
def status():
    """Get model status"""
    return jsonify({
        'model_loaded': model_loaded,
        'has_history': training_history is not None,
        'vocab_size_text': preprocessor.vocab_size_text if preprocessor else 0,
        'vocab_size_summary': preprocessor.vocab_size_summary if preprocessor else 0
    })


@app.route('/api/summarize', methods=['POST'])
def summarize():
    """Generate summary for input text"""
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 400

    data = request.get_json()
    input_text = data.get('text', '').strip()

    if not input_text:
        return jsonify({
            'success': False,
            'error': 'No input text provided'
        }), 400

    try:
        # Prepare input
        input_seq = preprocessor.prepare_input_for_prediction(input_text)

        # Generate summary
        reverse_target_word_index = preprocessor.get_reverse_word_index()
        target_word_index = preprocessor.get_word_index()

        summary = model.decode_sequence(
            input_seq,
            reverse_target_word_index,
            target_word_index,
            MAX_SUMMARY_LEN
        )

        return jsonify({
            'success': True,
            'summary': summary,
            'input_length': len(input_text.split()),
            'summary_length': len(summary.split())
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/training-history')
def get_training_history():
    """Get training history data"""
    if training_history is None:
        return jsonify({
            'success': False,
            'error': 'No training history available'
        }), 404

    # Calculate metrics
    final_train_loss = training_history['loss'][-1]
    final_val_loss = training_history['val_loss'][-1]
    final_train_acc = training_history['accuracy'][-1] * 100
    final_val_acc = training_history['val_accuracy'][-1] * 100
    best_epoch = training_history['val_loss'].index(min(training_history['val_loss'])) + 1

    return jsonify({
        'success': True,
        'epochs': list(range(1, len(training_history['loss']) + 1)),
        'train_loss': training_history['loss'],
        'val_loss': training_history['val_loss'],
        'train_acc': training_history['accuracy'],
        'val_acc': training_history['val_accuracy'],
        'metrics': {
            'final_train_loss': round(final_train_loss, 4),
            'final_val_loss': round(final_val_loss, 4),
            'final_train_acc': round(final_train_acc, 2),
            'final_val_acc': round(final_val_acc, 2),
            'best_epoch': best_epoch,
            'total_epochs': len(training_history['loss'])
        }
    })


@app.route('/api/model-info')
def get_model_info():
    """Get model architecture information"""
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 404

    return jsonify({
        'success': True,
        'architecture': 'Seq2Seq LSTM with Attention',
        'embedding_dim': 32,
        'latent_dim': 64,
        'max_text_len': MAX_TEXT_LEN,
        'max_summary_len': MAX_SUMMARY_LEN,
        'vocab_size_text': preprocessor.vocab_size_text,
        'vocab_size_summary': preprocessor.vocab_size_summary,
        'total_params': model.model.count_params()
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("LSTM TEXT SUMMARIZATION - WEB APPLICATION")
    print("="*70)
    print("\nLoading model and training history...")

    # Load model on startup
    model_status = load_model()
    history_status = load_training_history()

    print("\n" + "="*70)
    if model_status:
        print("✓ Model: LOADED")
    else:
        print("✗ Model: NOT FOUND - Please train first: python quick_demo_train.py")

    if history_status:
        print("✓ Training History: LOADED")
    else:
        print("✗ Training History: NOT FOUND")

    print("="*70)
    print("\n🚀 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5001)
