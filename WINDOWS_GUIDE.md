# 🪟 Windows Installation & Usage Guide

## Complete guide for running the Text Summarization Web App on Windows

---

## 📋 Prerequisites

1. **Python 3.7+** installed
   - Download from: https://www.python.org/downloads/
   - ✅ Check "Add Python to PATH" during installation!

2. **Git** (optional, for cloning)
   - Download from: https://git-scm.com/download/win

3. **Command Prompt** or **PowerShell**

---

## 🚀 Quick Start (Windows)

### Step 1: Setup Virtual Environment

```cmd
# Open Command Prompt or PowerShell
# Navigate to project directory
cd path\to\autosummarizeproject

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install tensorflow keras pandas numpy nltk flask matplotlib
```

### Step 2: Train the Model (2-3 minutes)

```cmd
# Make sure venv is activated (you should see (venv) in prompt)
python quick_demo_train.py
```

### Step 3: Launch Web App

**Option A: Using Batch File (Easiest)**
```cmd
# Double-click on: run_web.bat
# OR run in Command Prompt:
run_web.bat
```

**Option B: Manual Launch**
```cmd
# Activate virtual environment
venv\Scripts\activate

# Run the app
python app.py
```

### Step 4: Open Browser
```
http://localhost:5001
```

---

## 📁 Windows File Structure

```
autosummarizeproject\
├── venv\                      # Virtual environment (Windows)
│   └── Scripts\
│       ├── activate.bat       # Activation script
│       └── python.exe         # Python executable
├── app.py                     # Flask web server
├── model.py                   # Model architecture
├── train.py                   # Training script
├── quick_demo_train.py        # Fast training (2-3 min)
├── run_web.bat                # Windows launcher ⭐
├── run_web.sh                 # Mac/Linux launcher
├── templates\
│   └── index.html
├── static\
│   ├── css\
│   └── js\
└── model_weights.h5           # Trained model
```

---

## 🔧 Windows-Specific Commands

### Virtual Environment

**Activate:**
```cmd
venv\Scripts\activate
```

**Deactivate:**
```cmd
deactivate
```

**Check if activated:**
```cmd
# You should see (venv) at the start of your prompt
(venv) C:\Users\YourName\autosummarizeproject>
```

### Install Packages

```cmd
# Activate venv first!
venv\Scripts\activate

# Install all dependencies
pip install tensorflow keras pandas numpy nltk flask matplotlib

# Or use requirements file (if available)
pip install -r requirements.txt
```

### Run Training

```cmd
# Quick demo (2-3 minutes)
python quick_demo_train.py

# Standard training (longer, better quality)
python train.py

# Custom training
python train.py --sample_size 10000 --epochs 5
```

### Run Web App

```cmd
# Using batch file
run_web.bat

# OR manually
venv\Scripts\activate
python app.py
```

---

## 🎯 Complete Windows Setup (Step-by-Step)

### 1. Install Python

1. Download Python from: https://www.python.org/downloads/
2. Run installer
3. ✅ **IMPORTANT**: Check "Add Python to PATH"
4. Click "Install Now"
5. Verify installation:
   ```cmd
   python --version
   ```

### 2. Setup Project

```cmd
# Open Command Prompt (Win+R, type "cmd", press Enter)

# Navigate to project folder
cd C:\Users\YourName\Downloads\autosummarizeproject

# Create virtual environment
python -m venv venv

# Activate it (you'll see (venv) in prompt)
venv\Scripts\activate
```

### 3. Install Dependencies

```cmd
# Make sure (venv) is showing in your prompt!

# Install one by one:
pip install tensorflow
pip install keras
pip install pandas
pip install numpy
pip install nltk
pip install flask
pip install matplotlib

# OR install all at once:
pip install tensorflow keras pandas numpy nltk flask matplotlib
```

**Wait for installation to complete** (may take 5-10 minutes for TensorFlow)

### 4. Train Model

```cmd
# Still in activated venv
python quick_demo_train.py
```

**Expected output:**
```
======================================================================
ULTRA-FAST HOMEWORK DEMO MODEL (2-3 MINUTES)
======================================================================
...
Training... (wait 2-3 minutes)
...
✓ Training complete!
```

**Files created:**
- ✅ `model_weights.h5`
- ✅ `x_tokenizer.pickle`
- ✅ `y_tokenizer.pickle`
- ✅ `training_history.pickle`

### 5. Launch Web App

```cmd
# Double-click: run_web.bat
# OR in Command Prompt:
run_web.bat
```

**Expected output:**
```
==========================================
  Text Summarization Web App Launcher
==========================================

Activating virtual environment...
Starting web server...
Open your browser and go to: http://localhost:5001
...
```

### 6. Open Browser

- Open your browser (Chrome, Firefox, Edge)
- Go to: `http://localhost:5001`
- You should see the web interface!

---

## 🐛 Windows Troubleshooting

### Problem 1: "python is not recognized"

**Error:**
```
'python' is not recognized as an internal or external command
```

**Solution:**
1. Reinstall Python
2. ✅ Check "Add Python to PATH"
3. Or use full path: `C:\Python39\python.exe`

### Problem 2: "Access Denied" when creating venv

**Error:**
```
Access is denied
```

**Solution:**
```cmd
# Run Command Prompt as Administrator
# Right-click Command Prompt → "Run as administrator"
```

### Problem 3: "Scripts\activate cannot be loaded"

**Error (PowerShell):**
```
cannot be loaded because running scripts is disabled
```

**Solution:**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try again
venv\Scripts\activate
```

**Alternative:** Use Command Prompt instead of PowerShell

### Problem 4: Port 5001 already in use

**Error:**
```
Address already in use
```

**Solution:**
```cmd
# Find what's using the port
netstat -ano | findstr :5001

# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F

# OR edit app.py to use different port
# Change: app.run(port=5001)
# To: app.run(port=8080)
```

### Problem 5: TensorFlow installation fails

**Error:**
```
Could not find a version that satisfies tensorflow
```

**Solution:**
```cmd
# Try specific version
pip install tensorflow==2.13.0

# OR try CPU-only version
pip install tensorflow-cpu
```

### Problem 6: Module not found errors

**Error:**
```
ModuleNotFoundError: No module named 'flask'
```

**Solution:**
```cmd
# Make sure venv is activated (see (venv) in prompt)
venv\Scripts\activate

# Install missing module
pip install flask

# Install all dependencies
pip install tensorflow keras pandas numpy nltk flask matplotlib
```

### Problem 7: Browser can't connect

**Check:**
1. Is server running? (Check Command Prompt)
2. Correct URL? `http://localhost:5001` (not https)
3. Firewall blocking? (Allow Python in Windows Firewall)

**Solution:**
```cmd
# Try 127.0.0.1 instead of localhost
http://127.0.0.1:5001
```

---

## 📝 Windows Batch Files

### run_web.bat (Already created!)

**Usage:**
```cmd
# Method 1: Double-click file in Explorer
run_web.bat

# Method 2: Run in Command Prompt
cd path\to\project
run_web.bat
```

### Create Your Own Shortcuts

**Training Shortcut (train.bat):**
```batch
@echo off
echo Training model...
call venv\Scripts\activate
python quick_demo_train.py
pause
```

**Quick Launch (launch.bat):**
```batch
@echo off
call venv\Scripts\activate
start http://localhost:5001
python app.py
```

---

## 🎨 Windows-Specific Tips

### Tip 1: Create Desktop Shortcut

1. Right-click on `run_web.bat`
2. Select "Create shortcut"
3. Move shortcut to Desktop
4. Rename to "Text Summarizer"
5. Double-click to launch!

### Tip 2: Pin to Taskbar

1. Create shortcut to `run_web.bat`
2. Right-click shortcut
3. Select "Pin to taskbar"

### Tip 3: Auto-start browser

Edit `run_web.bat`, add before `python app.py`:
```batch
start http://localhost:5001
```

### Tip 4: Keep window open

Add at the end of `run_web.bat`:
```batch
pause
```

---

## 🔄 Daily Workflow (Windows)

**Morning routine:**
```cmd
# 1. Open project folder
cd C:\Users\YourName\autosummarizeproject

# 2. Double-click run_web.bat
# OR run in Command Prompt:
run_web.bat

# 3. Browser opens automatically to:
http://localhost:5001
```

**When done:**
- Press `Ctrl+C` in Command Prompt to stop server
- Close Command Prompt window

---

## 📊 Check Installation (Windows)

Run these commands to verify setup:

```cmd
# Check Python
python --version
# Should show: Python 3.x.x

# Check pip
pip --version
# Should show: pip x.x.x

# Check virtual environment exists
dir venv
# Should list folders: Include, Lib, Scripts

# Activate venv
venv\Scripts\activate

# Check packages installed
pip list
# Should show: tensorflow, flask, keras, etc.

# Check model files
dir *.h5
# Should show: model_weights.h5

# Check tokenizers
dir *.pickle
# Should show: x_tokenizer.pickle, y_tokenizer.pickle, training_history.pickle
```

---

## 🎯 Quick Commands Reference (Windows)

```cmd
# Setup (one-time)
python -m venv venv
venv\Scripts\activate
pip install tensorflow keras pandas numpy nltk flask matplotlib

# Train model (2-3 min)
python quick_demo_train.py

# Launch web app
run_web.bat
# OR
venv\Scripts\activate && python app.py

# Access in browser
http://localhost:5001

# Stop server
Ctrl+C

# Deactivate venv
deactivate
```

---

## 💾 Windows File Paths

**Using full paths in Windows:**
```cmd
# Activate venv with full path
C:\Users\YourName\autosummarizeproject\venv\Scripts\activate

# Run Python with full path
C:\Users\YourName\autosummarizeproject\venv\Scripts\python.exe app.py
```

**Using relative paths:**
```cmd
# From project folder
venv\Scripts\activate
python app.py
```

---

## 🌐 Accessing from Other Windows Devices

The web app is accessible from other devices on your network:

1. **Find your IP address:**
   ```cmd
   ipconfig
   ```
   Look for "IPv4 Address": e.g., `192.168.1.100`

2. **Access from other device:**
   ```
   http://192.168.1.100:5001
   ```

3. **Allow through Windows Firewall:**
   - Open "Windows Defender Firewall"
   - Click "Allow an app through firewall"
   - Find "Python" and check both Private and Public
   - Click OK

---

## 📚 Additional Resources

**Python on Windows:**
- https://www.python.org/downloads/windows/
- https://docs.python.org/3/using/windows.html

**Virtual Environments:**
- https://docs.python.org/3/library/venv.html

**TensorFlow on Windows:**
- https://www.tensorflow.org/install/pip#windows

---

## ✅ Verification Checklist

Before starting, verify:
- [ ] Python 3.7+ installed
- [ ] Python in PATH (run `python --version`)
- [ ] Virtual environment created (`venv\` folder exists)
- [ ] Dependencies installed (run `pip list`)
- [ ] Model trained (`.h5` file exists)
- [ ] Flask installed (check with `pip show flask`)
- [ ] Port 5001 available (not used by another app)

---

## 🎉 Success!

If everything works, you should see:

✅ Command Prompt shows server running
✅ Browser opens to web interface
✅ Can click example buttons
✅ Can generate summaries
✅ Can view training metrics

**Congratulations! Your text summarization system is running on Windows!** 🚀

---

## 📞 Need Help?

**Common issues:**
1. Check virtual environment is activated: `(venv)` in prompt
2. Check Python version: `python --version` (should be 3.7+)
3. Check dependencies: `pip list` (should show tensorflow, flask, etc.)
4. Check model files: `dir *.h5` (should show model_weights.h5)

**Still stuck?**
- Review error messages carefully
- Check troubleshooting section above
- Make sure all commands run in activated venv

---

**Windows guide complete!** Now just run `run_web.bat` and enjoy! 🎊
