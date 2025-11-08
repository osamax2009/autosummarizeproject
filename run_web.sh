#!/bin/bash

# Web Application Launcher for Text Summarization
# Starts the Flask web server

echo "=========================================="
echo "  Text Summarization Web App Launcher"
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

# Check if Flask is installed
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing Flask..."
    pip install flask
fi

# Check if model exists
if [ ! -f "model_weights.h5" ]; then
    echo ""
    echo "⚠️  Warning: Model not found!"
    echo "For best experience, train the model first:"
    echo "  python quick_demo_train.py"
    echo ""
    read -p "Press Enter to continue anyway..."
fi

# Launch Flask app
echo ""
echo "🚀 Starting web server..."
echo "📱 Open your browser and go to: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

python app.py
