#!/bin/bash
# Launcher script for the Text Summarization GUI
# This script tries different Python installations to find one that works

echo "Text Summarization GUI Launcher"
echo "================================"
echo ""

# Try to find a working Python installation
PYTHON_CMDS=(
    "/opt/homebrew/bin/python3.13"
    "/opt/homebrew/bin/python3"
    "/usr/local/bin/python3"
    "/usr/bin/python3"
    "python3"
    "python"
)

for CMD in "${PYTHON_CMDS[@]}"; do
    if command -v "$CMD" &> /dev/null; then
        echo "Testing: $CMD"

        # Check if required modules are available
        if $CMD -c "import tkinter, tensorflow, pandas, numpy" 2>/dev/null; then
            echo "✓ Found working Python: $CMD"
            echo ""
            echo "Starting GUI..."
            echo ""
            exec $CMD gui.py
        else
            echo "  Missing required modules"
        fi
    fi
done

echo ""
echo "ERROR: Could not find a Python installation with all required packages."
echo ""
echo "Please install the requirements using ONE of these methods:"
echo ""
echo "1. If you have a virtual environment, activate it:"
echo "   source venv/bin/activate"
echo "   python gui.py"
echo ""
echo "2. Install packages with pip (if allowed):"
echo "   pip3 install -r requirements.txt"
echo "   python3 gui.py"
echo ""
echo "3. Use the Python that trained your model (check setup/train logs)"
echo ""
