"""
Enhanced GUI Application for LSTM-based Text Summarization
With Training Charts, Accuracy Metrics, and Example Demos
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import pickle
from data_preprocessing import DataPreprocessor
from model import Seq2SeqLSTMSummarizer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')


class SummarizationGUI:
    """Enhanced GUI Application for text summarization"""

    def __init__(self, root):
        self.root = root
        self.root.title("LSTM Text Summarization - Enhanced with Metrics")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)

        # Model and preprocessor
        self.model = None
        self.preprocessor = None
        self.model_loaded = False

        # Configuration
        self.max_text_len = 100
        self.max_summary_len = 15

        # Training history for charts
        self.training_history = None

        # Example texts
        self.example_texts = [
            {
                "title": "Technology News",
                "text": "Apple announced its latest iPhone at a special event yesterday. The new device features improved cameras, faster processor, and longer battery life. The company CEO highlighted the enhanced AI capabilities and new design. Pre-orders start next week with delivery expected in two weeks. Analysts predict strong sales for the holiday season."
            },
            {
                "title": "Sports Update",
                "text": "The championship game ended with a dramatic finish last night. The home team scored in the final seconds to win by three points. Fans celebrated in the streets after the historic victory. The star player was named MVP of the game. This marks the team's first championship in over twenty years."
            },
            {
                "title": "Weather Report",
                "text": "A major storm system is approaching the coast bringing heavy rain and strong winds. Forecasters warn of potential flooding in low-lying areas. Residents are advised to prepare emergency supplies and stay indoors. The storm is expected to last through the weekend. Schools and businesses may close on Friday."
            }
        ]

        # Setup GUI
        self.setup_gui()

        # Try to load model on startup
        self.check_and_load_model()

        # Load training history if available
        self.load_training_history()

    def setup_gui(self):
        """Setup the GUI components with tabs"""

        # Configure colors
        self.bg_color = '#f0f0f0'
        self.header_color = '#2c3e50'

        self.root.configure(bg=self.bg_color)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Summarization
        self.summarization_tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.summarization_tab, text="📝 Summarization")

        # Tab 2: Training Metrics
        self.metrics_tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.metrics_tab, text="📊 Training Metrics")

        # Setup tabs
        self.setup_summarization_tab()
        self.setup_metrics_tab()

    def setup_summarization_tab(self):
        """Setup the summarization tab with examples"""
        parent = self.summarization_tab

        # Header
        header_frame = tk.Frame(parent, bg=self.header_color, height=70)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_label = tk.Label(
            header_frame,
            text="LSTM Text Summarization",
            font=('Arial', 20, 'bold'),
            bg=self.header_color,
            fg='white'
        )
        title_label.pack(pady=15)

        # Main container
        main_container = tk.Frame(parent, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Example selector frame
        example_frame = tk.LabelFrame(
            main_container,
            text="📚 Try Example Text",
            font=('Arial', 11, 'bold'),
            bg=self.bg_color,
            padx=10,
            pady=10
        )
        example_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            example_frame,
            text="Select an example to try:",
            font=('Arial', 10),
            bg=self.bg_color
        ).pack(side=tk.LEFT, padx=(0, 10))

        for example in self.example_texts:
            btn = tk.Button(
                example_frame,
                text=example["title"],
                command=lambda e=example: self.load_example(e),
                font=('Arial', 9),
                bg='#3498db',
                fg='white',
                relief=tk.FLAT,
                padx=12,
                pady=6,
                cursor='hand2'
            )
            btn.pack(side=tk.LEFT, padx=5)

        # Content container
        content_container = tk.Frame(main_container, bg=self.bg_color)
        content_container.pack(fill=tk.BOTH, expand=True)

        # Left Panel - Input
        left_panel = tk.LabelFrame(
            content_container,
            text="Input Text",
            font=('Arial', 12, 'bold'),
            bg=self.bg_color,
            padx=10,
            pady=10
        )
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Input text area
        self.input_text = scrolledtext.ScrolledText(
            left_panel,
            wrap=tk.WORD,
            font=('Arial', 11),
            height=20,
            bg='white',
            fg='black',
            relief=tk.SUNKEN,
            borderwidth=2,
            insertbackground='black'
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        self.input_text.insert("1.0", "Paste your text here or click an example above...")

        # Input buttons frame
        input_buttons_frame = tk.Frame(left_panel, bg=self.bg_color)
        input_buttons_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Button(
            input_buttons_frame,
            text="Load from File",
            command=self.load_text_from_file,
            font=('Arial', 10),
            bg='#95a5a6',
            fg='white',
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(
            input_buttons_frame,
            text="Clear",
            command=self.clear_input,
            font=('Arial', 10),
            bg='#e74c3c',
            fg='white',
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor='hand2'
        ).pack(side=tk.LEFT)

        # Right Panel - Output
        right_panel = tk.LabelFrame(
            content_container,
            text="Generated Summary",
            font=('Arial', 12, 'bold'),
            bg=self.bg_color,
            padx=10,
            pady=10
        )
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Output text area
        self.output_text = scrolledtext.ScrolledText(
            right_panel,
            wrap=tk.WORD,
            font=('Arial', 11),
            height=20,
            bg='#ecf0f1',
            fg='black',
            relief=tk.SUNKEN,
            borderwidth=2,
            insertbackground='black'
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.config(state=tk.DISABLED)

        # Output buttons frame
        output_buttons_frame = tk.Frame(right_panel, bg=self.bg_color)
        output_buttons_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Button(
            output_buttons_frame,
            text="Copy Summary",
            command=self.copy_summary,
            font=('Arial', 10),
            bg='#27ae60',
            fg='white',
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(
            output_buttons_frame,
            text="Save to File",
            command=self.save_summary,
            font=('Arial', 10),
            bg='#16a085',
            fg='white',
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor='hand2'
        ).pack(side=tk.LEFT)

        # Bottom control panel
        control_panel = tk.Frame(parent, bg=self.bg_color, height=100)
        control_panel.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(0, 15))

        # Summarize button
        self.summarize_btn = tk.Button(
            control_panel,
            text="Generate Summary",
            command=self.generate_summary,
            font=('Arial', 14, 'bold'),
            bg='#3498db',
            fg='white',
            relief=tk.FLAT,
            padx=30,
            pady=15,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.summarize_btn.pack(pady=10)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Loading model...")
        self.status_bar = tk.Label(
            control_panel,
            textvariable=self.status_var,
            font=('Arial', 10),
            bg=self.bg_color,
            fg='#7f8c8d'
        )
        self.status_bar.pack()

        # Progress bar
        self.progress = ttk.Progressbar(
            control_panel,
            mode='indeterminate',
            length=300
        )

    def setup_metrics_tab(self):
        """Setup the training metrics tab with charts"""
        parent = self.metrics_tab

        # Header
        header_frame = tk.Frame(parent, bg=self.header_color, height=70)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_label = tk.Label(
            header_frame,
            text="Training Metrics & Performance",
            font=('Arial', 20, 'bold'),
            bg=self.header_color,
            fg='white'
        )
        title_label.pack(pady=15)

        # Main container
        main_container = tk.Frame(parent, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Top info panel
        info_frame = tk.LabelFrame(
            main_container,
            text="Model Information",
            font=('Arial', 11, 'bold'),
            bg=self.bg_color,
            padx=15,
            pady=10
        )
        info_frame.pack(fill=tk.X, pady=(0, 15))

        self.model_info_text = tk.Text(
            info_frame,
            height=4,
            font=('Courier', 10),
            bg='#ecf0f1',
            fg='black',
            relief=tk.FLAT,
            borderwidth=0
        )
        self.model_info_text.pack(fill=tk.X)
        self.model_info_text.config(state=tk.DISABLED)

        # Charts container
        charts_frame = tk.Frame(main_container, bg=self.bg_color)
        charts_frame.pack(fill=tk.BOTH, expand=True)

        # Left chart - Loss
        left_chart_frame = tk.LabelFrame(
            charts_frame,
            text="Training & Validation Loss",
            font=('Arial', 11, 'bold'),
            bg=self.bg_color,
            padx=5,
            pady=5
        )
        left_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.loss_figure = Figure(figsize=(6, 4), dpi=100)
        self.loss_canvas = FigureCanvasTkAgg(self.loss_figure, left_chart_frame)
        self.loss_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right chart - Accuracy
        right_chart_frame = tk.LabelFrame(
            charts_frame,
            text="Training & Validation Accuracy",
            font=('Arial', 11, 'bold'),
            bg=self.bg_color,
            padx=5,
            pady=5
        )
        right_chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.acc_figure = Figure(figsize=(6, 4), dpi=100)
        self.acc_canvas = FigureCanvasTkAgg(self.acc_figure, right_chart_frame)
        self.acc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom metrics panel
        metrics_frame = tk.LabelFrame(
            main_container,
            text="Performance Summary",
            font=('Arial', 11, 'bold'),
            bg=self.bg_color,
            padx=15,
            pady=10
        )
        metrics_frame.pack(fill=tk.X, pady=(15, 0))

        self.metrics_text = tk.Text(
            metrics_frame,
            height=3,
            font=('Courier', 11, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50',
            relief=tk.FLAT,
            borderwidth=0
        )
        self.metrics_text.pack(fill=tk.X)
        self.metrics_text.config(state=tk.DISABLED)

    def load_example(self, example):
        """Load example text into input area"""
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", example["text"])
        self.status_var.set(f"Loaded example: {example['title']}")

    def load_training_history(self):
        """Load training history from pickle file"""
        history_path = 'training_history.pickle'
        if os.path.exists(history_path):
            try:
                with open(history_path, 'rb') as f:
                    self.training_history = pickle.load(f)
                self.update_metrics_display()
                self.status_var.set("Training history loaded successfully")
            except Exception as e:
                print(f"Could not load training history: {e}")
        else:
            self.show_no_history_message()

    def update_metrics_display(self):
        """Update the metrics tab with training history"""
        if self.training_history is None:
            self.show_no_history_message()
            return

        history = self.training_history

        # Update model info
        self.model_info_text.config(state=tk.NORMAL)
        self.model_info_text.delete("1.0", tk.END)
        info_text = f"""Model Architecture: Seq2Seq LSTM with Attention
Embedding Dimension: 32  |  Latent Dimension: 64  |  Max Text Length: 100  |  Max Summary Length: 15
Vocabulary Size (Text): {self.preprocessor.vocab_size_text if self.preprocessor else 'N/A'}  |  Vocabulary Size (Summary): {self.preprocessor.vocab_size_summary if self.preprocessor else 'N/A'}
Total Training Epochs: {len(history.get('loss', []))}"""
        self.model_info_text.insert("1.0", info_text)
        self.model_info_text.config(state=tk.DISABLED)

        # Plot Loss
        self.loss_figure.clear()
        ax1 = self.loss_figure.add_subplot(111)
        epochs = range(1, len(history['loss']) + 1)
        ax1.plot(epochs, history['loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
        ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax1.set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        self.loss_figure.tight_layout()
        self.loss_canvas.draw()

        # Plot Accuracy
        self.acc_figure.clear()
        ax2 = self.acc_figure.add_subplot(111)
        ax2.plot(epochs, history['accuracy'], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
        ax2.plot(epochs, history['val_accuracy'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax2.set_title('Model Accuracy Over Epochs', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        self.acc_figure.tight_layout()
        self.acc_canvas.draw()

        # Update metrics summary
        final_train_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        final_train_acc = history['accuracy'][-1] * 100
        final_val_acc = history['val_accuracy'][-1] * 100

        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        metrics_summary = f"""Final Training Loss: {final_train_loss:.4f}  |  Final Validation Loss: {final_val_loss:.4f}
Final Training Accuracy: {final_train_acc:.2f}%  |  Final Validation Accuracy: {final_val_acc:.2f}%
Best Epoch: {history['loss'].index(min(history['val_loss'])) + 1} (based on lowest validation loss)"""
        self.metrics_text.insert("1.0", metrics_summary)
        self.metrics_text.config(state=tk.DISABLED)

    def show_no_history_message(self):
        """Show message when no training history is available"""
        # Model info
        self.model_info_text.config(state=tk.NORMAL)
        self.model_info_text.delete("1.0", tk.END)
        self.model_info_text.insert("1.0", "No training history available. Train the model to see metrics here.")
        self.model_info_text.config(state=tk.DISABLED)

        # Show placeholder on charts
        self.loss_figure.clear()
        ax1 = self.loss_figure.add_subplot(111)
        ax1.text(0.5, 0.5, 'No training history available\nTrain the model first',
                ha='center', va='center', fontsize=14, color='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])
        self.loss_canvas.draw()

        self.acc_figure.clear()
        ax2 = self.acc_figure.add_subplot(111)
        ax2.text(0.5, 0.5, 'No training history available\nTrain the model first',
                ha='center', va='center', fontsize=14, color='gray')
        ax2.set_xticks([])
        ax2.set_yticks([])
        self.acc_canvas.draw()

        # Metrics summary
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert("1.0", "Train the model using: python quick_demo_train.py")
        self.metrics_text.config(state=tk.DISABLED)

    def check_and_load_model(self):
        """Check if model files exist and load them"""
        model_path = 'model_weights.h5'
        x_tokenizer_path = 'x_tokenizer.pickle'
        y_tokenizer_path = 'y_tokenizer.pickle'

        if os.path.exists(model_path) and os.path.exists(x_tokenizer_path) and os.path.exists(y_tokenizer_path):
            threading.Thread(target=self.load_model, daemon=True).start()
        else:
            self.status_var.set("Model not found. Please train the model first: python quick_demo_train.py")
            messagebox.showwarning(
                "Model Not Found",
                "Model files not found!\n\nPlease train the model first by running:\npython quick_demo_train.py\n\n"
                "This will train a demo model in 2-3 minutes."
            )

    def load_model(self):
        """Load the trained model and tokenizers"""
        try:
            self.status_var.set("Loading model and tokenizers...")

            # Load preprocessor
            self.preprocessor = DataPreprocessor(
                max_text_len=self.max_text_len,
                max_summary_len=self.max_summary_len
            )
            self.preprocessor.load_tokenizers()

            # Build and load model with correct parameters (matching quick_demo_train.py)
            self.model = Seq2SeqLSTMSummarizer(
                max_text_len=self.max_text_len,
                max_summary_len=self.max_summary_len,
                vocab_size_text=self.preprocessor.vocab_size_text,
                vocab_size_summary=self.preprocessor.vocab_size_summary,
                embedding_dim=32,   # MUST match trained model
                latent_dim=64       # MUST match trained model
            )

            self.model.build_model()
            self.model.load_weights('model_weights.h5')
            self.model.build_inference_models()

            self.model_loaded = True
            self.status_var.set("Model loaded successfully! Ready to summarize. Try an example above!")
            self.summarize_btn.config(state=tk.NORMAL, bg='#27ae60')

        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def generate_summary(self):
        """Generate summary from input text"""
        input_text = self.input_text.get("1.0", tk.END).strip()

        if not input_text or input_text == "Paste your text here or click an example above...":
            messagebox.showwarning("No Input", "Please enter some text to summarize.")
            return

        if not self.model_loaded:
            messagebox.showwarning("Model Not Loaded", "Model is not loaded yet. Please wait or train the model.")
            return

        # Disable button and show progress
        self.summarize_btn.config(state=tk.DISABLED, bg='#95a5a6')
        self.progress.pack(pady=5)
        self.progress.start(10)
        self.status_var.set("Generating summary...")
        self.root.update_idletasks()

        # Run summarization in background thread
        threading.Thread(target=self._generate_summary_thread, args=(input_text,), daemon=True).start()

    def _generate_summary_thread(self, input_text):
        """Background thread for summary generation"""
        try:
            # Prepare input
            input_seq = self.preprocessor.prepare_input_for_prediction(input_text)

            # Generate summary
            reverse_target_word_index = self.preprocessor.get_reverse_word_index()
            target_word_index = self.preprocessor.get_word_index()

            summary = self.model.decode_sequence(
                input_seq,
                reverse_target_word_index,
                target_word_index,
                self.max_summary_len
            )

            # Update output text
            self.root.after(0, self._update_output, summary)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, self._show_error, str(e))

    def _update_output(self, summary):
        """Update output text area with generated summary"""
        try:
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", summary)
            self.output_text.config(state=tk.DISABLED)

            # Re-enable button and hide progress
            self.progress.stop()
            self.progress.pack_forget()
            self.summarize_btn.config(state=tk.NORMAL, bg='#27ae60')
            self.status_var.set("Summary generated successfully!")

            self.root.update_idletasks()

        except Exception:
            import traceback
            traceback.print_exc()

    def _show_error(self, error_msg):
        """Show error message"""
        self.progress.stop()
        self.progress.pack_forget()
        self.summarize_btn.config(state=tk.NORMAL, bg='#27ae60')
        self.status_var.set(f"Error: {error_msg}")
        messagebox.showerror("Error", f"Failed to generate summary:\n{error_msg}")

    def load_text_from_file(self):
        """Load text from file"""
        try:
            self.root.update()
            file_path = filedialog.askopenfilename(
                parent=self.root,
                title="Select Text File",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )

            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    self.input_text.delete("1.0", tk.END)
                    self.input_text.insert("1.0", text)
                    self.status_var.set(f"Loaded text from: {os.path.basename(file_path)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
        except Exception:
            pass

    def clear_input(self):
        """Clear input text area"""
        self.input_text.delete("1.0", tk.END)
        self.status_var.set("Input cleared")

    def copy_summary(self):
        """Copy summary to clipboard"""
        summary = self.output_text.get("1.0", tk.END).strip()
        if summary:
            self.root.clipboard_clear()
            self.root.clipboard_append(summary)
            self.status_var.set("Summary copied to clipboard!")
        else:
            messagebox.showwarning("No Summary", "No summary to copy.")

    def save_summary(self):
        """Save summary to file"""
        summary = self.output_text.get("1.0", tk.END).strip()
        if not summary:
            messagebox.showwarning("No Summary", "No summary to save.")
            return

        try:
            self.root.update()
            file_path = filedialog.asksaveasfilename(
                parent=self.root,
                title="Save Summary",
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )

            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(summary)
                    self.status_var.set(f"Summary saved to: {os.path.basename(file_path)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")
        except Exception:
            pass


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = SummarizationGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
