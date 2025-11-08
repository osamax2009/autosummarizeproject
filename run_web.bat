@echo off
REM Web Application Launcher for Text Summarization (Windows)
REM Starts the Flask web server

echo ==========================================
echo   Text Summarization Web App Launcher
echo ==========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Error: Virtual environment not found!
    echo Please create it first with: python -m venv venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if Flask is installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing Flask...
    pip install flask
)

REM Check if model exists
if not exist "model_weights.h5" (
    echo.
    echo Warning: Model not found!
    echo For best experience, train the model first:
    echo   python quick_demo_train.py
    echo.
    pause
)

REM Launch Flask app
echo.
echo Starting web server...
echo Open your browser and go to: http://localhost:5001
echo.
echo Press Ctrl+C to stop the server
echo ==========================================
echo.

python app.py

pause
