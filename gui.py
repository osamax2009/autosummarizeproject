"""
GUI Application for LSTM-based Text Summarization
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
from data_preprocessing import DataPreprocessor
from model import Seq2SeqLSTMSummarizer
import warnings
warnings.filterwarnings('ignore')


class SummarizationGUI:
    """GUI Application for text summarization"""

    def __init__(self, root):
        self.root = root
        self.root.title("LSTM Text Summarization - CNN/DailyMail")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        # Model and preprocessor
        self.model = None
        self.preprocessor = None
        self.model_loaded = False

        # Configuration
        self.max_text_len = 200
        self.max_summary_len = 20

        # Setup GUI
        self.setup_gui()

        # Try to load model on startup
        self.check_and_load_model()

    def setup_gui(self):
        """Setup the GUI components"""

        # Configure style - use platform-appropriate theme
        style = ttk.Style()

        # Try to use the best available theme for the platform
        try:
            # On macOS, 'aqua' is native; on Windows, 'vista' or 'xpnative'; on Linux, 'clam'
            available_themes = style.theme_names()
            if 'aqua' in available_themes:
                style.theme_use('aqua')  # macOS native theme
            elif 'vista' in available_themes:
                style.theme_use('vista')  # Windows
            elif 'clam' in available_themes:
                style.theme_use('clam')  # Cross-platform
            else:
                # Fallback to whatever is available
                style.theme_use(available_themes[0])
        except Exception as e:
            print(f"Warning: Could not set theme: {e}")

        # Configure colors
        bg_color = '#f0f0f0'
        header_color = '#2c3e50'
        button_color = '#3498db'

        self.root.configure(bg=bg_color)

        # Header Frame
        header_frame = tk.Frame(self.root, bg=header_color, height=80)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_label = tk.Label(
            header_frame,
            text="LSTM Text Summarization",
            font=('Arial', 24, 'bold'),
            bg=header_color,
            fg='white'
        )
        title_label.pack(pady=20)

        # Main container
        main_container = tk.Frame(self.root, bg=bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left Panel - Input
        left_panel = tk.LabelFrame(
            main_container,
            text="Input Text",
            font=('Arial', 12, 'bold'),
            bg=bg_color,
            padx=10,
            pady=10
        )
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Instruction label
        instruction_label = tk.Label(
            left_panel,
            text="Paste or type your text here, or load from a file",
            font=('Arial', 9, 'italic'),
            bg=bg_color,
            fg='#7f8c8d'
        )
        instruction_label.pack(anchor=tk.W, pady=(0, 5))

        # Input text area
        self.input_text = scrolledtext.ScrolledText(
            left_panel,
            wrap=tk.WORD,
            font=('Arial', 11),
            height=20,
            bg='white',
            fg='black',  # Explicit text color
            relief=tk.SUNKEN,
            borderwidth=2,
            insertbackground='black'  # Cursor color
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        # Add placeholder text
        self.input_text.insert("1.0", "Paste your text here to generate a summary...")
        self.input_text.bind("<FocusIn>", self._clear_placeholder)

        # Input buttons frame
        input_buttons_frame = tk.Frame(left_panel, bg=bg_color)
        input_buttons_frame.pack(fill=tk.X, pady=(10, 0))

        self.load_file_btn = tk.Button(
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
        )
        self.load_file_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.clear_input_btn = tk.Button(
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
        )
        self.clear_input_btn.pack(side=tk.LEFT)

        # Right Panel - Output
        right_panel = tk.LabelFrame(
            main_container,
            text="Generated Summary",
            font=('Arial', 12, 'bold'),
            bg=bg_color,
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
            fg='black',  # Explicit text color
            relief=tk.SUNKEN,
            borderwidth=2,
            insertbackground='black'  # Cursor color
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.config(state=tk.DISABLED)

        # Output buttons frame
        output_buttons_frame = tk.Frame(right_panel, bg=bg_color)
        output_buttons_frame.pack(fill=tk.X, pady=(10, 0))

        self.copy_btn = tk.Button(
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
        )
        self.copy_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.save_btn = tk.Button(
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
        )
        self.save_btn.pack(side=tk.LEFT)

        # Bottom control panel
        control_panel = tk.Frame(self.root, bg=bg_color, height=100)
        control_panel.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(0, 20))

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
            bg=bg_color,
            fg='#7f8c8d'
        )
        self.status_bar.pack()

        # Progress bar
        self.progress = ttk.Progressbar(
            control_panel,
            mode='indeterminate',
            length=300
        )

    def _clear_placeholder(self, event=None):
        """Clear placeholder text when input field is focused"""
        _ = event  # event parameter required for bind callback
        current_text = self.input_text.get("1.0", tk.END).strip()
        if current_text == "Paste your text here to generate a summary...":
            self.input_text.delete("1.0", tk.END)
            self.input_text.unbind("<FocusIn>")  # Only clear once

    def check_and_load_model(self):
        """Check if model files exist and load them"""
        model_path = 'model_weights.h5'
        x_tokenizer_path = 'x_tokenizer.pickle'
        y_tokenizer_path = 'y_tokenizer.pickle'

        if os.path.exists(model_path) and os.path.exists(x_tokenizer_path) and os.path.exists(y_tokenizer_path):
            # Load in background thread
            threading.Thread(target=self.load_model, daemon=True).start()
        else:
            self.status_var.set("Model not found. Please train the model first by running: python train.py")
            messagebox.showwarning(
                "Model Not Found",
                "Model files not found!\n\nPlease train the model first by running:\npython train.py\n\n"
                "This will train the model using the CNN/DailyMail dataset."
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

            # Build and load model with MVP parameters (matches quick_train_mvp.py)
            self.model = Seq2SeqLSTMSummarizer(
                max_text_len=self.max_text_len,
                max_summary_len=self.max_summary_len,
                vocab_size_text=self.preprocessor.vocab_size_text,
                vocab_size_summary=self.preprocessor.vocab_size_summary,
                embedding_dim=64,   # MVP model uses 64 (not 128)
                latent_dim=128      # MVP model uses 128 (not 256)
            )

            self.model.build_model()
            self.model.load_weights('model_weights.h5')
            self.model.build_inference_models()

            self.model_loaded = True
            self.status_var.set("Model loaded successfully! Ready to summarize.")
            self.summarize_btn.config(state=tk.NORMAL, bg='#27ae60')

        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def generate_summary(self):
        """Generate summary from input text"""
        input_text = self.input_text.get("1.0", tk.END).strip()

        if not input_text:
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
        self.root.update_idletasks()  # Force UI update

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

            # Force UI update
            self.root.update_idletasks()

        except Exception:
            import traceback
            traceback.print_exc()

    def _show_error(self, error_msg):
        """Show error message"""
        self.progress.stop()
        self.progress.pack_forget()
        self.summarize_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Error: {error_msg}")
        messagebox.showerror("Error", f"Failed to generate summary:\n{error_msg}")

    def load_text_from_file(self):
        """Load text from file"""
        try:
            # Properly handle file dialog to avoid macOS warnings
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
        except Exception as e:
            # Silently handle dialog cancellation
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
            # Properly handle file dialog to avoid macOS warnings
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
        except Exception as e:
            # Silently handle dialog cancellation
            pass


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = SummarizationGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
