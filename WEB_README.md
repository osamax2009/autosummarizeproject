# 🌐 Web-Based LSTM Text Summarization

## Complete Mini-Model with HTML Interface

A practical, production-ready text summarization system with a modern web interface. This implementation includes all required features:

✅ **Training Metrics Visualization** - Interactive charts with Chart.js
✅ **Accuracy Display** - Real-time accuracy metrics
✅ **Example Text Library** - 3 pre-loaded examples for instant testing
✅ **Web Interface** - Modern HTML/CSS/JavaScript frontend
✅ **Python Backend** - Flask REST API
✅ **Mini-Model** - Fast training (2-3 minutes)

---

## 🚀 Super Quick Start (3 Steps)

```bash
# 1. Train the mini-model (2-3 minutes)
source venv/bin/activate
python quick_demo_train.py

# 2. Launch web app
./run_web.sh

# 3. Open browser
# Go to: http://localhost:5000
```

**That's it!** You now have a fully functional web-based summarization system.

---

## 📋 What You Get

### Web Interface Features

#### Tab 1: Summarization
- 🎯 **Example Text Buttons**: Click to load Technology, Sports, or Weather news
- 📝 **Input Panel**: Paste or type your text
- 🤖 **Generate Button**: One-click summarization
- 📊 **Word Counter**: Real-time word count
- 📋 **Copy Button**: Copy summary to clipboard
- ✨ **Modern UI**: Beautiful gradient design

#### Tab 2: Training Metrics
- 📈 **Loss Chart**: Interactive line chart (training vs validation)
- 📊 **Accuracy Chart**: Interactive line chart (training vs validation)
- 🏗️ **Model Info**: Architecture and configuration details
- 🎯 **Performance Cards**: Final metrics with gradient styling
- 🏆 **Best Epoch**: Identifies optimal training point

#### Tab 3: About
- 📚 Feature list
- 🏗️ Architecture details
- 📖 Usage instructions
- ℹ️ Training information

### Technical Stack

**Frontend:**
- HTML5 with modern semantic markup
- CSS3 with gradients and animations
- JavaScript ES6+ with async/await
- Chart.js 4.4.0 for visualizations

**Backend:**
- Flask (Python web framework)
- RESTful API architecture
- JSON data exchange
- Multi-threaded request handling

**ML Model:**
- Seq2Seq LSTM with Attention
- Bidirectional encoder
- 32-dim embeddings, 64-dim latent
- ~4M parameters

---

## 🎯 Complete Usage Guide

### Starting the Server

**Option 1: Launch Script** (Easiest)
```bash
./run_web.sh
```

**Option 2: Manual Start**
```bash
source venv/bin/activate
python app.py
```

### Using the Application

1. **Open Browser**: Navigate to `http://localhost:5000`

2. **Try an Example**:
   - Click "Technology News", "Sports Update", or "Weather Report"
   - Text loads automatically
   - Click "Generate Summary"
   - View AI-generated summary on the right

3. **Use Your Own Text**:
   - Clear the input area
   - Paste your article or story
   - Click "Generate Summary"
   - Wait 2-5 seconds for processing

4. **View Training Metrics**:
   - Click "Training Metrics" tab
   - See model architecture info
   - Explore interactive charts (hover for details)
   - Review performance summary

### Understanding the Charts

**Loss Chart (Lower is Better)**
- Blue line: How well model learns from training data
- Red line: How well model generalizes to new data
- Should both decrease over epochs
- Lines close together = good generalization
- Lines far apart = possible overfitting

**Accuracy Chart (Higher is Better)**
- Blue line: Training accuracy (percentage)
- Red line: Validation accuracy (percentage)
- Should both increase over epochs
- Final values show overall performance
- Hover over points for exact values

---

## 📱 Accessing from Other Devices

The server runs on `0.0.0.0`, making it accessible from:

### Same Computer
```
http://localhost:5000
http://127.0.0.1:5000
```

### Other Devices on Same Network
```
http://YOUR_IP_ADDRESS:5000
```

**Find Your IP:**
```bash
# macOS/Linux
ifconfig | grep "inet " | grep -v 127.0.0.1

# Or use
hostname -I
```

**Example**: If your IP is `192.168.1.100`, access via:
```
http://192.168.1.100:5000
```

---

## 🎨 Screenshots Description

### Home Page
- Purple gradient header with "Model Ready" badge
- Three tabs: Summarization, Training Metrics, About
- Example text buttons below header
- Side-by-side input/output panels
- Large "Generate Summary" button
- Success/error alerts

### Summarization Tab
- Left panel: Input text area with word count
- Right panel: Generated summary with word count
- Clear button, Copy button
- Real-time status updates
- Smooth animations

### Training Metrics Tab
- Top: Model information grid (8 cards)
- Middle: Two side-by-side charts
  - Left: Loss chart (blue/red lines)
  - Right: Accuracy chart (blue/red lines)
- Bottom: Purple gradient metrics summary
- 6 metric cards showing final values

---

## 🔧 Customization Guide

### Change Port Number
Edit `app.py` (last line):
```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Change 5000 to 8080
```

### Add New Examples
Edit `app.py`:
```python
EXAMPLE_TEXTS.append({
    "id": "science",
    "title": "Science News",
    "text": "Your example text here..."
})
```

### Change Color Scheme
Edit `static/css/style.css`:
```css
/* Header gradient */
.header {
    background: linear-gradient(135deg, #YOUR_COLOR1 0%, #YOUR_COLOR2 100%);
}

/* Primary button */
.btn-primary {
    background: linear-gradient(135deg, #YOUR_COLOR3 0%, #YOUR_COLOR4 100%);
}
```

### Modify Model Parameters
If you retrain with different parameters, update `app.py`:
```python
MAX_TEXT_LEN = 200      # Increase for longer inputs
MAX_SUMMARY_LEN = 30    # Increase for longer summaries
```

---

## 📊 API Documentation

### Endpoints

#### `GET /`
Returns main HTML page

#### `GET /api/status`
**Response:**
```json
{
  "model_loaded": true,
  "has_history": true,
  "vocab_size_text": 34085,
  "vocab_size_summary": 10960
}
```

#### `POST /api/summarize`
**Request:**
```json
{
  "text": "Your article text here..."
}
```

**Success Response:**
```json
{
  "success": true,
  "summary": "AI generated summary...",
  "input_length": 75,
  "summary_length": 12
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Model not loaded"
}
```

#### `GET /api/training-history`
Returns complete training metrics with epochs, losses, accuracies

#### `GET /api/model-info`
Returns model architecture and configuration details

---

## 🐛 Troubleshooting

### Server Won't Start

**Error**: `Address already in use`
```bash
# Find process using port 5000
lsof -i :5000

# Kill it
kill -9 <PID>

# Or change port in app.py
```

**Error**: `Flask module not found`
```bash
source venv/bin/activate
pip install flask
```

### Model Not Loading

**Error**: `Model not found`
```bash
# Train the model first
python quick_demo_train.py

# Should create:
# - model_weights.h5
# - x_tokenizer.pickle
# - y_tokenizer.pickle
# - training_history.pickle
```

### Charts Not Showing

**Cause**: No training history file

**Solution**:
```bash
# Remove old model files
rm model_weights.h5 training_history.pickle

# Retrain to generate history
python quick_demo_train.py
```

### Browser Can't Connect

**Check**:
1. Is server running? (Look for startup messages)
2. Correct URL? (`http://localhost:5000`)
3. Firewall blocking port 5000?
4. Using `http://` not `https://`?

### Slow Summarization

**Normal**: First summary takes 5-10 seconds (model loading)

**Subsequent**: Should be 2-3 seconds

**If Very Slow**:
- Check CPU usage
- Close other applications
- Restart server

---

## 🚀 Production Deployment

### For Production Use:

1. **Disable Debug Mode**
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

2. **Use Production Server**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

3. **Add HTTPS**
```bash
# Using Let's Encrypt + Nginx
# Or use cloud platform (Heroku, Railway, etc.)
```

4. **Environment Variables**
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key
```

### Cloud Platforms

**Heroku**:
1. Create `Procfile`: `web: gunicorn app:app`
2. Create `requirements.txt`
3. `git push heroku main`

**Railway/Render**:
1. Connect GitHub repo
2. Set start command: `python app.py`
3. Auto-deploy on push

---

## 📈 Performance Specs

### Model
- **Training Time**: 2-3 minutes (demo model)
- **Inference Time**: 2-5 seconds per summary
- **Memory Usage**: ~500MB RAM
- **CPU Usage**: Moderate (no GPU needed)

### Web Server
- **Concurrent Users**: 10-20 (with gunicorn)
- **Response Time**: <3 seconds (after warmup)
- **Supported Browsers**: Chrome, Firefox, Safari, Edge (modern versions)

### Scalability
- **Vertical**: Increase RAM/CPU
- **Horizontal**: Use load balancer + multiple instances
- **Caching**: Add Redis for frequent requests

---

## 🎓 Learning Resources

### Understanding the Code

**Flask Routes** (`app.py`):
- `/` → Serves HTML page
- `/api/*` → JSON API endpoints

**HTML Template** (`templates/index.html`):
- Jinja2 templating
- Dynamic content rendering

**JavaScript** (`static/js/app.js`):
- Fetch API for AJAX requests
- Chart.js for visualizations
- DOM manipulation

**CSS** (`static/css/style.css`):
- Flexbox and Grid layouts
- CSS animations
- Gradient styling

### Extending the Project

**Add User Authentication**:
```python
from flask_login import LoginManager, login_required

@app.route('/api/summarize')
@login_required
def summarize():
    # ...
```

**Add Database**:
```python
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///summaries.db'
db = SQLAlchemy(app)
```

**Add File Upload**:
```html
<input type="file" id="fileUpload" accept=".txt">
```

---

## 📚 Additional Documentation

- **[WEB_APP_GUIDE.md](WEB_APP_GUIDE.md)** - Detailed guide
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start for desktop GUI
- **[CHANGES.md](CHANGES.md)** - All enhancements and fixes
- **[README.md](README.md)** - Main project documentation

---

## 🎉 Summary

You now have a **complete, production-ready web application** for text summarization:

✅ Beautiful HTML interface
✅ Interactive training charts
✅ Real-time accuracy metrics
✅ Example text library
✅ RESTful API
✅ Mobile-responsive design
✅ Professional styling
✅ Complete documentation

**Launch it**: `./run_web.sh`
**Access it**: `http://localhost:5000`
**Enjoy it**: Start summarizing! 🚀

---

**Questions?** Check the troubleshooting section or [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md)

**Happy Summarizing!** 🎊
