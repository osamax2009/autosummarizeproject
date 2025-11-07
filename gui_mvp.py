"""
GUI for MVP Model (with correct embedding_dim=64, latent_dim=128)
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
        self.root.title("LSTM Text Summarization - MVP Demo")
        self.root.geometry("1000x700")

        # MVP Model parameters
        self.max_text_len = 200
        self.max_summary_len = 20
        self.embedding_dim = 64   # MVP uses 64
        self.latent_dim = 128     # MVP uses 128

        self.model = None
        self.preprocessor = None
        self.model_loaded = False

        self.setup_gui()
        self.check_and_load_model()

    def setup_gui(self):
        """Setup GUI components"""
        style = ttk.Style()
        try:
            available_themes = style.theme_names()
            if 'aqua' in available_themes:
                style.theme_use('aqua')
            elif 'clam' in available_themes:
                style.theme_use('clam')
        except:
            pass

        bg_color = '#f0f0f0'
        header_color = '#2c3e50'

        self.root.configure(bg=bg_color)

        # Header
        header = tk.Frame(self.root, bg=header_color, height=80)
        header.pack(fill=tk.X, side=tk.TOP)
        tk.Label(header, text="LSTM Text Summarization - MVP Demo",
                font=('Arial', 24, 'bold'), bg=header_color, fg='white').pack(pady=20)

        # Main container
        main = tk.Frame(self.root, bg=bg_color)
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left panel
        left = tk.LabelFrame(main, text="Input Text", font=('Arial', 12, 'bold'),
                            bg=bg_color, padx=10, pady=10)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(left, text="Paste or type your text here", font=('Arial', 9, 'italic'),
                bg=bg_color, fg='#7f8c8d').pack(anchor=tk.W, pady=(0, 5))

        self.input_text = scrolledtext.ScrolledText(left, wrap=tk.WORD, font=('Arial', 11),
                                                    height=20, bg='white', fg='black',
                                                    relief=tk.SUNKEN, borderwidth=2)
        self.input_text.pack(fill=tk.BOTH, expand=True)
        self.input_text.insert("1.0", "Paste your text here...")
        self.input_text.bind("<FocusIn>", lambda e: self.input_text.delete("1.0", tk.END)
                            if "Paste your text" in self.input_text.get("1.0", tk.END) else None)

        btn_frame = tk.Frame(left, bg=bg_color)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Button(btn_frame, text="Clear", command=lambda: self.input_text.delete("1.0", tk.END),
                 font=('Arial', 10), bg='#e74c3c', fg='white', relief=tk.FLAT,
                 padx=15, pady=8).pack(side=tk.LEFT)

        # Right panel
        right = tk.LabelFrame(main, text="Generated Summary", font=('Arial', 12, 'bold'),
                             bg=bg_color, padx=10, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(right, wrap=tk.WORD, font=('Arial', 11),
                                                     height=20, bg='#ecf0f1', fg='black',
                                                     relief=tk.SUNKEN, borderwidth=2)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.config(state=tk.DISABLED)

        # Bottom control
        control = tk.Frame(self.root, bg=bg_color)
        control.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(0, 20))

        self.summarize_btn = tk.Button(control, text="Generate Summary",
                                       command=self.generate_summary,
                                       font=('Arial', 14, 'bold'), bg='#3498db', fg='white',
                                       relief=tk.FLAT, padx=30, pady=15, state=tk.DISABLED)
        self.summarize_btn.pack(pady=10)

        self.status_var = tk.StringVar(value="Loading model...")
        tk.Label(control, textvariable=self.status_var, font=('Arial', 10),
                bg=bg_color, fg='#7f8c8d').pack()

        self.progress = ttk.Progressbar(control, mode='indeterminate', length=300)

    def check_and_load_model(self):
        """Load model in background"""
        if os.path.exists('model_weights.h5'):
            threading.Thread(target=self.load_model, daemon=True).start()
        else:
            self.status_var.set("Model not found. Run: python quick_train_mvp.py")

    def load_model(self):
        """Load the model"""
        try:
            self.status_var.set("Loading model...")

            self.preprocessor = DataPreprocessor(
                max_text_len=self.max_text_len,
                max_summary_len=self.max_summary_len
            )
            self.preprocessor.load_tokenizers()

            self.model = Seq2SeqLSTMSummarizer(
                max_text_len=self.max_text_len,
                max_summary_len=self.max_summary_len,
                vocab_size_text=self.preprocessor.vocab_size_text,
                vocab_size_summary=self.preprocessor.vocab_size_summary,
                embedding_dim=self.embedding_dim,  # MVP: 64
                latent_dim=self.latent_dim         # MVP: 128
            )

            self.model.build_model()
            self.model.load_weights('model_weights.h5')
            self.model.build_inference_models()

            self.model_loaded = True
            self.status_var.set("✓ Model loaded! Ready to summarize.")
            self.summarize_btn.config(state=tk.NORMAL, bg='#27ae60')

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def generate_summary(self):
        """Generate summary"""
        text = self.input_text.get("1.0", tk.END).strip()

        if not text or "Paste your text" in text:
            messagebox.showwarning("No Input", "Please enter text to summarize.")
            return

        if not self.model_loaded:
            messagebox.showwarning("Model Not Ready", "Model is still loading...")
            return

        self.summarize_btn.config(state=tk.DISABLED, bg='#95a5a6')
        self.progress.pack(pady=5)
        self.progress.start(10)
        self.status_var.set("Generating summary...")
        self.root.update_idletasks()

        threading.Thread(target=self._generate_thread, args=(text,), daemon=True).start()

    def _generate_thread(self, text):
        """Background summary generation"""
        try:
            input_seq = self.preprocessor.prepare_input_for_prediction(text)
            reverse_idx = self.preprocessor.get_reverse_word_index()
            word_idx = self.preprocessor.get_word_index()

            summary = self.model.decode_sequence(input_seq, reverse_idx, word_idx,
                                                self.max_summary_len)

            self.root.after(0, self._update_output, summary)
        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _update_output(self, summary):
        """Update output"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", summary)
        self.output_text.config(state=tk.DISABLED)

        self.progress.stop()
        self.progress.pack_forget()
        self.summarize_btn.config(state=tk.NORMAL, bg='#27ae60')
        self.status_var.set("✓ Summary generated!")
        self.root.update_idletasks()

    def _show_error(self, msg):
        """Show error"""
        self.progress.stop()
        self.progress.pack_forget()
        self.summarize_btn.config(state=tk.NORMAL, bg='#27ae60')
        self.status_var.set(f"Error: {msg}")
        messagebox.showerror("Error", f"Failed to generate summary:\n{msg}")


if __name__ == '__main__':
    root = tk.Tk()
    app = SummarizationGUI(root)
    root.mainloop()
