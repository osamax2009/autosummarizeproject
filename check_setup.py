"""
Setup Checker - Verify installation and environment
"""

import sys
import os


def check_python_version():
    """Check Python version"""
    print("\n[1/6] Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"    ✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"    ✗ Python {version.major}.{version.minor}.{version.micro} (Requires 3.7+)")
        return False


def check_packages():
    """Check required packages"""
    print("\n[2/6] Checking required packages...")
    packages = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'nltk': 'NLTK',
        'sklearn': 'scikit-learn'
    }

    all_installed = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"    ✓ {name} (installed)")
        except ImportError:
            print(f"    ✗ {name} (NOT installed)")
            all_installed = False

    if not all_installed:
        print("\n    Install missing packages with:")
        print("    pip install -r requirements.txt")

    return all_installed


def check_tensorflow_gpu():
    """Check if TensorFlow can access GPU"""
    print("\n[3/6] Checking GPU availability...")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"    ✓ GPU available ({len(gpus)} device(s))")
            for i, gpu in enumerate(gpus):
                print(f"      - GPU {i}: {gpu.name}")
            return True
        else:
            print("    ⚠ No GPU detected (training will use CPU)")
            return True
    except Exception as e:
        print(f"    ✗ Error checking GPU: {str(e)}")
        return False


def check_dataset_files():
    """Check if dataset files exist"""
    print("\n[4/6] Checking dataset files...")
    files = {
        'train.csv': 'Training data',
        'validation.csv': 'Validation data',
        'test.csv': 'Test data'
    }

    all_exist = True
    for file, desc in files.items():
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
            print(f"    ✓ {desc} ({size:.1f} MB)")
        else:
            print(f"    ✗ {desc} (NOT found)")
            all_exist = False

    return all_exist


def check_model_files():
    """Check if trained model files exist"""
    print("\n[5/6] Checking trained model files...")
    files = {
        'model_weights.h5': 'Model weights',
        'x_tokenizer.pickle': 'Input tokenizer',
        'y_tokenizer.pickle': 'Output tokenizer'
    }

    all_exist = True
    for file, desc in files.items():
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
            print(f"    ✓ {desc} ({size:.1f} MB)")
        else:
            print(f"    ⚠ {desc} (not trained yet)")
            all_exist = False

    if not all_exist:
        print("\n    Train the model with:")
        print("    python quick_train.py  (fast, for testing)")
        print("    or")
        print("    python train.py  (full training)")

    return all_exist


def check_tkinter():
    """Check if tkinter is available for GUI"""
    print("\n[6/6] Checking GUI support (tkinter)...")
    try:
        import tkinter
        print("    ✓ tkinter available (GUI will work)")
        return True
    except ImportError:
        print("    ✗ tkinter NOT available (GUI won't work)")
        print("\n    Install tkinter:")
        print("    - Ubuntu/Debian: sudo apt-get install python3-tk")
        print("    - macOS: brew install python-tk")
        print("    - Windows: Usually included with Python")
        return False


def print_summary(results):
    """Print summary of checks"""
    print("\n" + "="*70)
    print("SETUP CHECK SUMMARY")
    print("="*70)

    all_ready = all(results.values())

    if all_ready:
        print("\n✓ All checks passed! Your environment is ready.")
        if results['model']:
            print("\n  You can now run:")
            print("  python gui.py")
        else:
            print("\n  Next step: Train the model")
            print("  python quick_train.py")
    else:
        print("\n⚠ Some issues detected. Please resolve them before proceeding.")

        if not results['python']:
            print("\n  Critical: Update Python to version 3.7 or higher")

        if not results['packages']:
            print("\n  Required: Install missing packages")
            print("  pip install -r requirements.txt")

        if not results['dataset']:
            print("\n  Required: Dataset files missing")
            print("  Ensure train.csv, validation.csv, and test.csv are present")

        if not results['tkinter']:
            print("\n  Optional: Install tkinter for GUI support")

    print("\n" + "="*70)


def main():
    """Main function"""
    print("="*70)
    print("LSTM TEXT SUMMARIZATION - SETUP CHECKER")
    print("="*70)
    print("\nVerifying your environment and installation...")

    results = {
        'python': check_python_version(),
        'packages': check_packages(),
        'gpu': check_tensorflow_gpu(),
        'dataset': check_dataset_files(),
        'model': check_model_files(),
        'tkinter': check_tkinter()
    }

    print_summary(results)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCheck interrupted by user.")
    except Exception as e:
        print(f"\n\nError during setup check: {str(e)}")
