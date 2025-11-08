#!/bin/bash

# Run GUI Script for Text Summarization
# This script activates the virtual environment and launches the GUI

echo "=========================================="
echo "  Text Summarization GUI Launcher"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found!"
    echo "Please create it first with: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if matplotlib is installed
python -c "import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing matplotlib..."
    pip install matplotlib
fi

# Check if model exists
if [ ! -f "model_weights.h5" ]; then
    echo ""
    echo "⚠️  Warning: Model not found!"
    echo "Please train the model first by running:"
    echo "  python quick_demo_train.py"
    echo ""
    read -p "Press Enter to continue anyway (GUI will show warning)..."
fi

# Launch GUI
echo ""
echo "🚀 Launching Enhanced GUI..."
echo ""
python gui.py

echo ""
echo "✅ GUI closed."
