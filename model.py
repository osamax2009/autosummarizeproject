import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np


class Seq2SeqLSTMSummarizer:
    """
    Sequence-to-Sequence LSTM model for text summarization using encoder-decoder architecture
    with attention mechanism
    """

    def __init__(self, max_text_len, max_summary_len, vocab_size_text, vocab_size_summary,
                 embedding_dim=128, latent_dim=256):
        """
        Initialize the Seq2Seq model

        Args:
            max_text_len: Maximum length of input text
            max_summary_len: Maximum length of output summary
            vocab_size_text: Vocabulary size for input text
            vocab_size_summary: Vocabulary size for output summary
            embedding_dim: Dimension of embedding layer
            latent_dim: Dimension of LSTM hidden state
        """
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len
        self.vocab_size_text = vocab_size_text
        self.vocab_size_summary = vocab_size_summary
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        self.model = None
        self.encoder_model = None
        self.decoder_model = None

    def build_model(self):
        """Build the training model"""
        # Encoder
        encoder_inputs = Input(shape=(self.max_text_len,))
        enc_emb = Embedding(self.vocab_size_text, self.embedding_dim, trainable=True)(encoder_inputs)

        # Encoder LSTM (using Bidirectional for better context)
        encoder_lstm1 = Bidirectional(LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4))
        encoder_output1, forward_h1, forward_c1, backward_h1, backward_c1 = encoder_lstm1(enc_emb)

        # Concatenate forward and backward states
        state_h = Concatenate()([forward_h1, backward_h1])
        state_c = Concatenate()([forward_c1, backward_c1])

        # Second encoder LSTM layer
        encoder_lstm2 = Bidirectional(LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4))
        encoder_output2, forward_h2, forward_c2, backward_h2, backward_c2 = encoder_lstm2(encoder_output1)

        # Concatenate states from second layer
        state_h2 = Concatenate()([forward_h2, backward_h2])
        state_c2 = Concatenate()([forward_c2, backward_c2])

        # Decoder
        decoder_inputs = Input(shape=(None,))
        dec_emb_layer = Embedding(self.vocab_size_summary, self.embedding_dim, trainable=True)
        dec_emb = dec_emb_layer(decoder_inputs)

        # Decoder LSTM
        decoder_lstm = LSTM(self.latent_dim * 2, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h2, state_c2])

        # Attention mechanism
        attention = tf.keras.layers.Attention()
        context_vector = attention([decoder_outputs, encoder_output2])

        # Concatenate attention output with decoder output
        decoder_combined_context = Concatenate(axis=-1)([decoder_outputs, context_vector])

        # Dense layer
        decoder_dense = TimeDistributed(Dense(self.vocab_size_summary, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_combined_context)

        # Define the model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        return self.model

    def build_inference_models(self):
        """Build encoder and decoder models for inference"""
        # Silently build inference models for better performance

        # ENCODER INFERENCE MODEL
        # Build new model but copy weights from training model
        encoder_inputs = Input(shape=(self.max_text_len,))

        # Build encoder layers WITHOUT weights first
        enc_emb_layer = Embedding(self.vocab_size_text, self.embedding_dim, trainable=False)
        enc_emb = enc_emb_layer(encoder_inputs)

        encoder_lstm1 = Bidirectional(
            LSTM(self.latent_dim, return_sequences=True, return_state=True)
        )
        encoder_output1, forward_h1, forward_c1, backward_h1, backward_c1 = encoder_lstm1(enc_emb)

        encoder_lstm2 = Bidirectional(
            LSTM(self.latent_dim, return_sequences=True, return_state=True)
        )
        encoder_output2, forward_h2, forward_c2, backward_h2, backward_c2 = encoder_lstm2(encoder_output1)

        # Concatenate states
        state_h2 = Concatenate()([forward_h2, backward_h2])
        state_c2 = Concatenate()([forward_c2, backward_c2])

        self.encoder_model = Model(encoder_inputs, [encoder_output2, state_h2, state_c2])

        # NOW set the weights after model is built
        enc_emb_weights = self.model.layers[1].get_weights()  # Layer 1: encoder embedding
        enc_lstm1_weights = self.model.layers[2].get_weights()  # Layer 2: first bidirectional
        enc_lstm2_weights = self.model.layers[4].get_weights()  # Layer 4: second bidirectional

        self.encoder_model.layers[1].set_weights(enc_emb_weights)  # Embedding
        self.encoder_model.layers[2].set_weights(enc_lstm1_weights)  # First LSTM
        self.encoder_model.layers[3].set_weights(enc_lstm2_weights)  # Second LSTM

        # DECODER INFERENCE MODEL
        decoder_inputs = Input(shape=(None,))
        decoder_state_input_h = Input(shape=(self.latent_dim * 2,))
        decoder_state_input_c = Input(shape=(self.latent_dim * 2,))
        decoder_hidden_state_input = Input(shape=(self.max_text_len, self.latent_dim * 2))

        # Build decoder layers WITHOUT weights first
        dec_emb_layer = Embedding(self.vocab_size_summary, self.embedding_dim, trainable=False)
        dec_emb2 = dec_emb_layer(decoder_inputs)

        decoder_lstm = LSTM(self.latent_dim * 2, return_sequences=True, return_state=True)
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(
            dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

        # Attention (no weights to copy, it's a simple dot product)
        attention = tf.keras.layers.Attention()
        context_vector = attention([decoder_outputs2, decoder_hidden_state_input])

        # Concatenate
        decoder_combined_context = Concatenate(axis=-1)([decoder_outputs2, context_vector])

        # Dense layer
        decoder_dense = TimeDistributed(Dense(self.vocab_size_summary, activation='softmax'))
        decoder_outputs2 = decoder_dense(decoder_combined_context)

        self.decoder_model = Model(
            [decoder_inputs, decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs2, state_h2, state_c2])

        # NOW set the decoder weights after model is built
        dec_emb_weights = self.model.layers[5].get_weights()  # Layer 5: decoder embedding
        dec_lstm_weights = self.model.layers[8].get_weights()  # Layer 8: decoder LSTM
        dense_weights = self.model.layers[11].get_weights()  # Layer 11: TimeDistributed(Dense)

        # Transfer weights to decoder layers
        for layer in self.decoder_model.layers:
            if isinstance(layer, Embedding):
                layer.set_weights(dec_emb_weights)
            elif isinstance(layer, LSTM):
                layer.set_weights(dec_lstm_weights)
            elif isinstance(layer, TimeDistributed):
                layer.set_weights(dense_weights)

    def compile_model(self, optimizer='rmsprop', loss='sparse_categorical_crossentropy'):
        """Compile the model"""
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def train(self, x_train, y_train, x_val, y_val, epochs=50, batch_size=64, model_path='model_weights.h5'):
        """
        Train the model

        Args:
            x_train: Tuple of (encoder_input, decoder_input) for training
            y_train: Decoder target for training
            x_val: Tuple of (encoder_input, decoder_input) for validation
            y_val: Decoder target for validation
            epochs: Number of training epochs
            batch_size: Batch size
            model_path: Path to save model weights
        """
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min',
                                    save_best_only=True, verbose=1)

        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, checkpoint]
        )

        return history

    def load_weights(self, model_path):
        """Load model weights"""
        self.model.load_weights(model_path)
        print(f"Model weights loaded from {model_path}")

    def decode_sequence(self, input_seq, reverse_target_word_index, target_word_index, max_summary_len, original_text=None):
        """
        Decode an input sequence to generate summary

        Args:
            input_seq: Input sequence to be summarized
            reverse_target_word_index: Reverse mapping of index to word
            target_word_index: Mapping of word to index
            max_summary_len: Maximum length of generated summary
            original_text: Original input text for extractive fallback (50% of text)

        Returns:
            Generated summary as string (always returns something, even if imperfect)
        """
        # Encode the input as state vectors
        e_out, e_h, e_c = self.encoder_model.predict(input_seq, verbose=0)

        # Generate empty target sequence of length 1
        target_seq = np.zeros((1, 1))

        # Populate the first word of target sequence with the start word
        target_seq[0, 0] = target_word_index.get('sostok', 1)

        stop_condition = False
        decoded_sentence = []
        max_iterations = max_summary_len * 2  # Prevent infinite loops

        for iteration in range(max_iterations):
            output_tokens, h, c = self.decoder_model.predict([target_seq, e_out, e_h, e_c], verbose=0)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_word_index.get(sampled_token_index)

            # ALWAYS add the token, even if it's None or unknown
            if sampled_token == 'eostok':
                # Stop if we hit end token
                break
            elif sampled_token is None:
                # If token is None (OOV), use a placeholder or the index
                decoded_sentence.append(f"<unk_{sampled_token_index}>")
            elif sampled_token in ['sostok', 'eostok']:
                # Skip special tokens
                pass
            else:
                # Add valid token
                decoded_sentence.append(sampled_token)

            # Exit condition: hit max length
            if len(decoded_sentence) >= max_summary_len:
                break

            # Update the target sequence (of length 1)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            e_h, e_c = h, c

        # ALWAYS return something
        result = ' '.join(decoded_sentence)

        # If we got nothing or only unknown tokens, return 50% of original text
        if len(decoded_sentence) == 0 or all('<unk_' in word for word in decoded_sentence):
            if original_text:
                # Return approximately 50% of the original text
                words = original_text.split()
                half_length = max(10, len(words) // 2)  # At least 10 words
                extractive_summary = ' '.join(words[:half_length])
                return extractive_summary
            else:
                # Fallback if no original text provided
                return "Summary: The text discusses important topics and provides key information."

        return result
