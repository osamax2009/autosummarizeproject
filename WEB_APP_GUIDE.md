# Web Application Guide - LSTM Text Summarization

## 🌟 Overview

This is a modern, web-based interface for the LSTM text summarization model. Built with Flask, HTML, CSS, and JavaScript, it provides a professional and user-friendly experience with interactive training metrics visualization.

## ✨ Features

### 🎯 Summarization Tab
- **Example Text Library**: 3 pre-loaded examples (Technology, Sports, Weather)
- **Real-time Summarization**: Generate summaries with one click
- **Word Count**: Live word counter for input and output
- **Copy to Clipboard**: Easy copy functionality
- **Responsive Design**: Works on desktop, tablet, and mobile

### 📊 Training Metrics Tab
- **Interactive Charts**: Powered by Chart.js
  - Loss chart (training vs validation)
  - Accuracy chart (training vs validation)
- **Model Information**: Detailed architecture specs
- **Performance Summary**: Final metrics and best epoch
- **Visual Metrics Cards**: Beautiful gradient-styled metrics display

### ℹ️ About Tab
- Model architecture details
- How to use instructions
- Feature list
- Training information

## 🚀 Quick Start

### Option 1: Using Launch Script (Recommended)

```bash
# Make it executable (first time only)
chmod +x run_web.sh

# Launch the web app
./run_web.sh
```

### Option 2: Manual Launch

```bash
# Activate virtual environment
source venv/bin/activate

# Install Flask (if not already installed)
pip install flask

# Run the application
python app.py
```

### Option 3: One-Line Launch

```bash
source venv/bin/activate && python app.py
```

## 📱 Accessing the Application

Once the server starts, open your browser and navigate to:

```
http://localhost:5000
```

Or from another device on the same network:

```
http://YOUR_IP_ADDRESS:5000
```

To find your IP address:
- **Mac/Linux**: `ifconfig | grep "inet "`
- **Windows**: `ipconfig`

## 💡 How to Use

### 1. Generating Summaries

**Method A: Using Examples**
1. Click on any example button (Technology News, Sports Update, Weather Report)
2. The text will automatically load into the input area
3. Click "Generate Summary"
4. View the AI-generated summary on the right

**Method B: Custom Text**
1. Paste or type your text in the input area
2. Click "Generate Summary"
3. Wait a few seconds for processing
4. View your summary on the right
5. Click "Copy" to copy the summary to clipboard

### 2. Viewing Training Metrics

1. Click on the "Training Metrics" tab
2. View model information at the top
3. See two interactive charts:
   - **Left**: Training vs Validation Loss
   - **Right**: Training vs Validation Accuracy
4. Review performance summary at the bottom with key metrics

### 3. Understanding the Charts

**Loss Chart**:
- **Blue line**: Training loss (should decrease)
- **Red line**: Validation loss (should decrease)
- Lower is better
- Hover over points to see exact values

**Accuracy Chart**:
- **Blue line**: Training accuracy (should increase)
- **Red line**: Validation accuracy (should increase)
- Higher is better (percentage)
- Hover over points to see exact values

## 🎨 Interface Features

### Status Badge
- **Gray pulsing**: Loading
- **Green**: Model ready
- **Red**: Model not loaded

### Alerts
- **Green**: Success messages
- **Red**: Error messages
- Auto-dismiss after 5 seconds

### Responsive Layout
- Desktop: Side-by-side panels
- Mobile: Stacked panels
- Charts adjust to screen size

## 📊 API Endpoints

The application provides a RESTful API:

### GET `/api/status`
Returns model status and configuration

```json
{
  "model_loaded": true,
  "has_history": true,
  "vocab_size_text": 34085,
  "vocab_size_summary": 10960
}
```

### POST `/api/summarize`
Generate summary for input text

**Request:**
```json
{
  "text": "Your text here..."
}
```

**Response:**
```json
{
  "success": true,
  "summary": "Generated summary...",
  "input_length": 50,
  "summary_length": 10
}
```

### GET `/api/training-history`
Get training metrics data

**Response:**
```json
{
  "success": true,
  "epochs": [1, 2, 3],
  "train_loss": [2.5, 2.1, 1.8],
  "val_loss": [2.6, 2.2, 1.9],
  "train_acc": [0.45, 0.52, 0.58],
  "val_acc": [0.44, 0.51, 0.57],
  "metrics": {
    "final_train_loss": 1.8,
    "final_val_loss": 1.9,
    "final_train_acc": 58.0,
    "final_val_acc": 57.0,
    "best_epoch": 3,
    "total_epochs": 3
  }
}
```

### GET `/api/model-info`
Get model architecture details

**Response:**
```json
{
  "success": true,
  "architecture": "Seq2Seq LSTM with Attention",
  "embedding_dim": 32,
  "latent_dim": 64,
  "max_text_len": 100,
  "max_summary_len": 15,
  "vocab_size_text": 34085,
  "vocab_size_summary": 10960,
  "total_params": 4123456
}
```

## 🔧 Configuration

Edit `app.py` to change:

```python
# Port number
app.run(port=5000)  # Change to any available port

# Host (for external access)
app.run(host='0.0.0.0')  # Allows access from network

# Debug mode
app.run(debug=True)  # Set to False for production
```

## 📁 File Structure

```
project/
├── app.py                    # Flask application
├── templates/
│   └── index.html           # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css        # Styling
│   └── js/
│       └── app.js           # JavaScript logic
├── model.py                 # Model architecture
├── data_preprocessing.py    # Data utilities
├── model_weights.h5         # Trained model
├── x_tokenizer.pickle       # Text tokenizer
├── y_tokenizer.pickle       # Summary tokenizer
└── training_history.pickle  # Training metrics
```

## 🐛 Troubleshooting

### "Model Not Loaded" Error
**Cause**: Model files not found
**Solution**:
```bash
python quick_demo_train.py
```

### Port Already in Use
**Cause**: Another application using port 5000
**Solution**: Change port in app.py or kill the process:
```bash
# Find process
lsof -i :5000

# Kill process
kill -9 <PID>
```

### Charts Not Showing
**Cause**: Training history not available
**Solution**: Retrain model to generate history:
```bash
python quick_demo_train.py
```

### Cannot Access from Other Devices
**Cause**: Firewall or host setting
**Solution**:
1. Ensure `app.run(host='0.0.0.0')` in app.py
2. Check firewall allows port 5000
3. Use correct IP address

## 🎯 Best Practices

### For Best Results:
1. **Input Text**: 50-200 words works best
2. **Content Type**: News-style articles perform better
3. **Complete Sentences**: Avoid fragments or keywords only

### Performance:
- First summary may take longer (model loading)
- Subsequent summaries are faster
- Process runs on CPU (GPU not required for inference)

## 🌐 Deployment Options

### Local Network
Already configured for local network access with `host='0.0.0.0'`

### Cloud Deployment (Basic)
1. **Heroku**:
   - Create `Procfile`: `web: python app.py`
   - Create `requirements.txt` with all dependencies
   - Deploy: `git push heroku main`

2. **Railway/Render**:
   - Connect GitHub repository
   - Set start command: `python app.py`
   - Deploy automatically

3. **Docker**:
   - Create `Dockerfile`
   - Build: `docker build -t summarizer .`
   - Run: `docker run -p 5000:5000 summarizer`

## 🎨 Customization

### Change Colors
Edit `static/css/style.css`:
```css
/* Header gradient */
background: linear-gradient(135deg, #YOUR_COLOR1 0%, #YOUR_COLOR2 100%);

/* Button colors */
background: #YOUR_COLOR;
```

### Add New Examples
Edit `app.py`:
```python
EXAMPLE_TEXTS.append({
    "id": "custom",
    "title": "Your Title",
    "text": "Your example text..."
})
```

### Modify Model Parameters
Edit `app.py`:
```python
MAX_TEXT_LEN = 100      # Change max input length
MAX_SUMMARY_LEN = 15    # Change max summary length
```

## 📚 Technologies Used

- **Backend**: Flask (Python web framework)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Charts**: Chart.js 4.4.0
- **ML**: TensorFlow/Keras
- **Design**: Custom CSS with gradient themes

## 🔒 Security Notes

### For Production:
1. Set `debug=False` in app.py
2. Use environment variables for secrets
3. Add input validation and sanitization
4. Implement rate limiting
5. Use HTTPS with SSL certificate
6. Add authentication if needed

## 💬 Support

### Common Questions

**Q: Can I use my own model?**
A: Yes! Replace `model_weights.h5` and tokenizer files, update parameters in app.py

**Q: How do I change the port?**
A: Edit the last line of app.py: `app.run(port=YOUR_PORT)`

**Q: Can this run on Windows?**
A: Yes! Use `python app.py` directly or create a `.bat` file

**Q: How much RAM does it need?**
A: Minimum 2GB, 4GB recommended for smooth operation

**Q: Is GPU required?**
A: No, CPU is sufficient for inference (GPU helps with training)

## 🎉 Enjoy!

Your modern web-based text summarization system is ready to use. The interface provides a professional, user-friendly experience with beautiful visualizations and real-time summarization capabilities.

Access it at: **http://localhost:5000**

---

**Happy Summarizing!** 🚀
