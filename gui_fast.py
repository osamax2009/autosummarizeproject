"""
FAST & RESPONSIVE GUI for Homework Demo
Optimized for quick response and better appearance
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import os
from data_preprocessing import DataPreprocessor
from model import Seq2SeqLSTMSummarizer
import warnings
warnings.filterwarnings('ignore')


class FastSummarizationGUI:
    """Optimized GUI for text summarization"""

    def __init__(self, root):
        self.root = root
        self.root.title("AI Text Summarizer - Homework Demo")
        self.root.geometry("1100x750")

        # Model state
        self.model = None
        self.preprocessor = None
        self.model_loaded = False
        self.is_generating = False

        # Setup GUI first
        self.setup_gui()

        # Load model in background
        threading.Thread(target=self.load_model, daemon=True).start()

    def setup_gui(self):
        """Setup modern, responsive GUI"""

        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # Colors
        bg = '#f5f5f5'
        primary = '#2196F3'
        success = '#4CAF50'
        self.root.configure(bg=bg)

        # Header
        header = tk.Frame(self.root, bg=primary, height=70)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="🤖 LSTM Text Summarizer",
            font=('Helvetica', 20, 'bold'),
            bg=primary,
            fg='white'
        ).pack(pady=20)

        # Main container
        main = tk.Frame(self.root, bg=bg)
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Left - Input
        left = tk.LabelFrame(main, text="📝 Input Text", font=('Helvetica', 11, 'bold'), bg=bg, padx=10, pady=10)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.input_text = scrolledtext.ScrolledText(
            left, wrap=tk.WORD, font=('Helvetica', 10),
            height=20, bg='white', relief=tk.SOLID, borderwidth=1
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        self.input_text.insert("1.0", "Paste your text here and click Generate Summary...")
        self.input_text.bind("<Button-1>", lambda e: self.input_text.delete("1.0", tk.END) if "Paste your text" in self.input_text.get("1.0", tk.END) else None)

        # Input buttons
        btn_frame = tk.Frame(left, bg=bg)
        btn_frame.pack(fill=tk.X, pady=(8, 0))

        tk.Button(
            btn_frame, text="Clear", command=lambda: self.input_text.delete("1.0", tk.END),
            font=('Helvetica', 9), bg='#FF5252', fg='white', relief=tk.FLAT, padx=12, pady=6
        ).pack(side=tk.LEFT)

        # Right - Output
        right = tk.LabelFrame(main, text="✨ Generated Summary", font=('Helvetica', 11, 'bold'), bg=bg, padx=10, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(
            right, wrap=tk.WORD, font=('Helvetica', 10),
            height=20, bg='#e8f5e9', relief=tk.SOLID, borderwidth=1
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.config(state=tk.DISABLED)

        # Output buttons
        out_btn_frame = tk.Frame(right, bg=bg)
        out_btn_frame.pack(fill=tk.X, pady=(8, 0))

        tk.Button(
            out_btn_frame, text="📋 Copy", command=self.copy_summary,
            font=('Helvetica', 9), bg='#2196F3', fg='white', relief=tk.FLAT, padx=12, pady=6
        ).pack(side=tk.LEFT, padx=(0, 5))

        # Bottom control
        control = tk.Frame(self.root, bg=bg)
        control.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Generate button
        self.gen_btn = tk.Button(
            control,
            text="⚡ Generate Summary",
            command=self.generate_summary,
            font=('Helvetica', 13, 'bold'),
            bg=success,
            fg='white',
            relief=tk.FLAT,
            padx=40,
            pady=12,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.gen_btn.pack(pady=10)

        # Status
        self.status = tk.Label(
            control,
            text="⏳ Loading model...",
            font=('Helvetica', 10),
            bg=bg,
            fg='#666'
        )
        self.status.pack()

        # Progress
        self.progress = ttk.Progressbar(control, mode='indeterminate', length=400)

    def load_model(self):
        """Load model in background"""
        try:
            self.status.config(text="⏳ Loading tokenizers...")
            self.root.update_idletasks()

            self.preprocessor = DataPreprocessor(max_text_len=200, max_summary_len=20)
            self.preprocessor.load_tokenizers()

            self.status.config(text="⏳ Building model...")
            self.root.update_idletasks()

            self.model = Seq2SeqLSTMSummarizer(
                max_text_len=200,
                max_summary_len=20,
                vocab_size_text=self.preprocessor.vocab_size_text,
                vocab_size_summary=self.preprocessor.vocab_size_summary,
                embedding_dim=128,
                latent_dim=256
            )

            self.model.build_model()
            self.model.load_weights('model_weights.h5')

            self.status.config(text="⏳ Preparing inference...")
            self.root.update_idletasks()

            self.model.build_inference_models()

            self.model_loaded = True
            self.status.config(text="✅ Ready! Paste text and click Generate Summary", fg='#4CAF50')
            self.gen_btn.config(state=tk.NORMAL, bg='#4CAF50')

        except Exception as e:
            self.status.config(text=f"❌ Error: {str(e)[:50]}...", fg='#f44336')
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def generate_summary(self):
        """Generate summary with optimized response"""

        if self.is_generating:
            return

        text = self.input_text.get("1.0", tk.END).strip()

        if not text or "Paste your text" in text:
            messagebox.showwarning("No Input", "Please enter some text!")
            return

        if not self.model_loaded:
            messagebox.showwarning("Not Ready", "Model is still loading...")
            return

        # Update UI immediately
        self.is_generating = True
        self.gen_btn.config(state=tk.DISABLED, bg='#9E9E9E', text="⏳ Generating...")
        self.progress.pack(pady=5)
        self.progress.start(10)
        self.status.config(text="🔄 Generating summary...", fg='#FF9800')
        self.root.update_idletasks()

        # Generate in background
        threading.Thread(target=self._generate, args=(text,), daemon=True).start()

    def _generate(self, text):
        """Background generation"""
        try:
            # Truncate long text for faster processing
            words = text.split()
            if len(words) > 150:
                text = ' '.join(words[:150])

            input_seq = self.preprocessor.prepare_input_for_prediction(text)
            reverse_idx = self.preprocessor.get_reverse_word_index()
            target_idx = self.preprocessor.get_word_index()

            summary = self.model.decode_sequence(input_seq, reverse_idx, target_idx, 20)

            self.root.after(0, self._update_result, summary)

        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _update_result(self, summary):
        """Update with result"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)

        if '<OOV>' in summary or not summary.strip():
            self.output_text.insert("1.0", "⚠️ Model needs more training.\nTry running with more data.")
        else:
            self.output_text.insert("1.0", summary)

        self.output_text.config(state=tk.DISABLED)

        self.progress.stop()
        self.progress.pack_forget()
        self.gen_btn.config(state=tk.NORMAL, bg='#4CAF50', text="⚡ Generate Summary")
        self.status.config(text="✅ Summary generated!", fg='#4CAF50')
        self.is_generating = False

    def _show_error(self, error):
        """Show error"""
        self.progress.stop()
        self.progress.pack_forget()
        self.gen_btn.config(state=tk.NORMAL, bg='#4CAF50', text="⚡ Generate Summary")
        self.status.config(text=f"❌ Error: {error[:40]}...", fg='#f44336')
        self.is_generating = False
        messagebox.showerror("Error", error)

    def copy_summary(self):
        """Copy to clipboard"""
        summary = self.output_text.get("1.0", tk.END).strip()
        if summary:
            self.root.clipboard_clear()
            self.root.clipboard_append(summary)
            self.status.config(text="✅ Copied to clipboard!", fg='#4CAF50')


def main():
    root = tk.Tk()
    app = FastSummarizationGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
