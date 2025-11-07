import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Data preprocessing class for text summarization
    """

    def __init__(self, max_text_len=200, max_summary_len=20, vocab_size=10000):
        """
        Initialize the preprocessor

        Args:
            max_text_len: Maximum length of input text
            max_summary_len: Maximum length of summary
            vocab_size: Maximum vocabulary size
        """
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len
        self.vocab_size = vocab_size

        self.x_tokenizer = None
        self.y_tokenizer = None

    def clean_text(self, text, remove_stopwords=False):
        """
        Clean text by removing special characters, extra spaces, etc.

        Args:
            text: Input text string
            remove_stopwords: Whether to remove stopwords

        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r"[^a-zA-Z?.!,]+", " ", text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove stopwords if specified
        if remove_stopwords:
            try:
                stop_words = set(stopwords.words('english'))
                text = ' '.join([word for word in text.split() if word not in stop_words])
            except:
                # If NLTK stopwords not available, skip this step
                pass

        return text.strip()

    def load_data(self, file_path, sample_size=None):
        """
        Load data from CSV file

        Args:
            file_path: Path to CSV file
            sample_size: Number of samples to load (None for all)

        Returns:
            DataFrame with article and highlights
        """
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)

        if sample_size:
            df = df.head(sample_size)

        print(f"Loaded {len(df)} samples")
        return df

    def preprocess_data(self, df, clean_stopwords=False):
        """
        Preprocess the dataset

        Args:
            df: DataFrame with 'article' and 'highlights' columns
            clean_stopwords: Whether to remove stopwords

        Returns:
            DataFrame with cleaned text
        """
        print("Preprocessing data...")

        # Clean articles and highlights
        df['cleaned_text'] = df['article'].apply(lambda x: self.clean_text(x, clean_stopwords))
        df['cleaned_summary'] = df['highlights'].apply(lambda x: self.clean_text(x, clean_stopwords))

        # Add start and end tokens to summary
        df['cleaned_summary'] = df['cleaned_summary'].apply(lambda x: 'sostok ' + x + ' eostok')

        # Remove empty entries
        df = df[df['cleaned_text'] != '']
        df = df[df['cleaned_summary'] != 'sostok  eostok']

        # Drop rows with very short text or summary
        df = df[df['cleaned_text'].apply(lambda x: len(x.split()) > 5)]
        df = df[df['cleaned_summary'].apply(lambda x: len(x.split()) > 2)]

        print(f"Data preprocessed. {len(df)} samples remaining")
        return df

    def prepare_tokenizers(self, text_data, summary_data):
        """
        Prepare tokenizers for text and summary

        Args:
            text_data: List of text strings
            summary_data: List of summary strings
        """
        print("Preparing tokenizers...")

        # Tokenizer for text
        self.x_tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.x_tokenizer.fit_on_texts(list(text_data))

        # Tokenizer for summary
        self.y_tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.y_tokenizer.fit_on_texts(list(summary_data))

        # Calculate actual vocab sizes
        self.vocab_size_text = len(self.x_tokenizer.word_index) + 1
        self.vocab_size_summary = len(self.y_tokenizer.word_index) + 1

        print(f"Text vocabulary size: {self.vocab_size_text}")
        print(f"Summary vocabulary size: {self.vocab_size_summary}")

    def texts_to_sequences(self, text_data, summary_data):
        """
        Convert text and summary to sequences

        Args:
            text_data: List of text strings
            summary_data: List of summary strings

        Returns:
            Padded sequences for text and summary
        """
        print("Converting texts to sequences...")

        # Convert to sequences
        x_sequences = self.x_tokenizer.texts_to_sequences(text_data)
        y_sequences = self.y_tokenizer.texts_to_sequences(summary_data)

        # Pad sequences
        x_padded = pad_sequences(x_sequences, maxlen=self.max_text_len, padding='post')
        y_padded = pad_sequences(y_sequences, maxlen=self.max_summary_len, padding='post')

        print(f"Text sequences shape: {x_padded.shape}")
        print(f"Summary sequences shape: {y_padded.shape}")

        return x_padded, y_padded

    def prepare_decoder_input_output(self, y_padded):
        """
        Prepare decoder input and output sequences

        Args:
            y_padded: Padded summary sequences

        Returns:
            Decoder input and output sequences
        """
        # Decoder input: all tokens except last
        decoder_input = y_padded[:, :-1]

        # Decoder output: all tokens except first
        decoder_output = y_padded[:, 1:]

        # Reshape decoder output for sparse categorical crossentropy
        decoder_output = decoder_output.reshape(decoder_output.shape[0], decoder_output.shape[1], 1)

        return decoder_input, decoder_output

    def save_tokenizers(self, x_tokenizer_path='x_tokenizer.pickle', y_tokenizer_path='y_tokenizer.pickle'):
        """
        Save tokenizers to files

        Args:
            x_tokenizer_path: Path to save text tokenizer
            y_tokenizer_path: Path to save summary tokenizer
        """
        with open(x_tokenizer_path, 'wb') as f:
            pickle.dump(self.x_tokenizer, f)

        with open(y_tokenizer_path, 'wb') as f:
            pickle.dump(self.y_tokenizer, f)

        print(f"Tokenizers saved to {x_tokenizer_path} and {y_tokenizer_path}")

    def load_tokenizers(self, x_tokenizer_path='x_tokenizer.pickle', y_tokenizer_path='y_tokenizer.pickle'):
        """
        Load tokenizers from files

        Args:
            x_tokenizer_path: Path to text tokenizer
            y_tokenizer_path: Path to summary tokenizer
        """
        with open(x_tokenizer_path, 'rb') as f:
            self.x_tokenizer = pickle.load(f)

        with open(y_tokenizer_path, 'rb') as f:
            self.y_tokenizer = pickle.load(f)

        self.vocab_size_text = len(self.x_tokenizer.word_index) + 1
        self.vocab_size_summary = len(self.y_tokenizer.word_index) + 1

        print(f"Tokenizers loaded. Text vocab: {self.vocab_size_text}, Summary vocab: {self.vocab_size_summary}")

    def prepare_input_for_prediction(self, text):
        """
        Prepare input text for prediction

        Args:
            text: Input text string

        Returns:
            Padded sequence ready for model input
        """
        cleaned_text = self.clean_text(text)
        sequence = self.x_tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_text_len, padding='post')
        return padded_sequence

    def get_reverse_word_index(self):
        """
        Get reverse word index for summary tokenizer

        Returns:
            Dictionary mapping index to word
        """
        reverse_target_word_index = self.y_tokenizer.index_word
        return reverse_target_word_index

    def get_word_index(self):
        """
        Get word index for summary tokenizer

        Returns:
            Dictionary mapping word to index
        """
        return self.y_tokenizer.word_index


def prepare_training_data(train_path, val_path, sample_size=None, max_text_len=200, max_summary_len=20):
    """
    Complete pipeline to prepare training data

    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        sample_size: Number of samples to use (None for all)
        max_text_len: Maximum length of input text
        max_summary_len: Maximum length of summary

    Returns:
        Tuple of (preprocessor, training data, validation data)
    """
    preprocessor = DataPreprocessor(max_text_len=max_text_len, max_summary_len=max_summary_len)

    # Load and preprocess training data
    train_df = preprocessor.load_data(train_path, sample_size)
    train_df = preprocessor.preprocess_data(train_df)

    # Load and preprocess validation data
    val_df = preprocessor.load_data(val_path, sample_size // 5 if sample_size else None)
    val_df = preprocessor.preprocess_data(val_df)

    # Prepare tokenizers
    preprocessor.prepare_tokenizers(
        pd.concat([train_df['cleaned_text'], val_df['cleaned_text']]),
        pd.concat([train_df['cleaned_summary'], val_df['cleaned_summary']])
    )

    # Convert to sequences
    x_train, y_train = preprocessor.texts_to_sequences(
        train_df['cleaned_text'], train_df['cleaned_summary'])
    x_val, y_val = preprocessor.texts_to_sequences(
        val_df['cleaned_text'], val_df['cleaned_summary'])

    # Prepare decoder input and output
    decoder_input_train, decoder_output_train = preprocessor.prepare_decoder_input_output(y_train)
    decoder_input_val, decoder_output_val = preprocessor.prepare_decoder_input_output(y_val)

    # Save tokenizers
    preprocessor.save_tokenizers()

    training_data = {
        'encoder_input': x_train,
        'decoder_input': decoder_input_train,
        'decoder_output': decoder_output_train
    }

    validation_data = {
        'encoder_input': x_val,
        'decoder_input': decoder_input_val,
        'decoder_output': decoder_output_val
    }

    return preprocessor, training_data, validation_data
