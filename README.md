# 📊 AI Data Analyst

> **Professional AI-powered data analysis tool** that allows users to upload data files and ask questions in plain English.

## 🚀 Features

- 📁 **Universal File Support**: CSV, Excel, PDF, Word documents
- 💬 **Modern Chat Interface**: ChatGPT-like conversational experience
- 📊 **Real-time Dashboard**: Instant data metrics and insights
- ⚡ **Smart Sampling**: Fast analysis for large datasets
- 🎨 **Professional UI**: Custom theme and responsive design
- 🛡️ **Robust Error Handling**: Graceful failure recovery

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python with LangChain
- **AI Models**: OpenRouter (Meta Llama, Gemini)
- **Data Processing**: Pandas, pdfplumber, python-docx
- **Deployment**: Render (Free Tier Optimized)

## 🌐 Live Demo

[![Deploy on Render](https://render.com/images/badge.svg)](https://render.com/deploy?button=true)

**Deployed URL**: `https://my-data-analyst.onrender.com`

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenRouter API Key

### Local Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/ai-data-analyst.git
cd ai-data-analyst

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🔧 Configuration

### Environment Variables
```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your_api_key_here"
```

### Theme Configuration
The app uses a professional blue theme configured in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#0068C9"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 🚀 Deployment Pipeline

### Step 1: GitHub Repository
1. Create `.gitignore` file with:
   ```
   .venv/
   __pycache__/
   .env
   .streamlit/secrets.toml
   *.DS_Store
   ```

2. Initialize and push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/ai-data-analyst.git
   git push -u origin main
   ```

### Step 2: Render Deployment
1. Go to [dashboard.render.com](https://dashboard.render.com)
2. **New + → Web Service**
3. Connect GitHub repository
4. Configure settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py`
   - **Plan**: Free
5. **Environment Variables**:
   - **Key**: `OPENROUTER_API_KEY`
   - **Value**: Your OpenRouter API key
6. **Create Web Service**

### Step 3: Free Tier Optimization
The app is optimized for Render's free tier:
- **Lightweight Libraries**: pdfplumber instead of docling
- **Memory Efficient**: < 512MB RAM usage
- **Fast Startup**: Optimized imports and caching

## 📱 Usage Guide

### 1. Upload Data
- Supported formats: CSV, Excel, PDF, Word
- Maximum file size: 200MB
- Automatic table detection

### 2. Configure Settings
- **Model Selection**: Meta Llama 3.1 8B (default) or Gemini 2.0 Flash
- **Performance**: Fast Mode for large datasets
- **Output**: Simple explanations or professional terminology

### 3. Ask Questions
Examples:
- "What are the total sales by product line?"
- "Show me the top 10 customers by revenue"
- "What is the average order value?"
- "Find all transactions over $1000"

### 4. Get Results
- Instant AI-powered analysis
- Formatted Markdown tables
- Export capabilities
- Chat history preservation

## 🎯 Performance Features

### Smart Caching
- **Data Loading**: `@st.cache_data` for fast file loading
- **Model Responses**: Reduced API calls with caching
- **UI Components**: Efficient rendering

### Fast Mode
- **Data Sampling**: 20% sample for instant analysis
- **Row Limits**: Configurable display limits
- **Memory Optimization**: Large dataset handling

### Error Recovery
- **Parsing Errors**: Smart data extraction from failures
- **API Failures**: Graceful fallbacks
- **File Errors**: User-friendly error messages

## 🔒 Security

- **API Keys**: Stored in environment variables
- **No Data Persistence**: Files processed in memory only
- **Secure Deployment**: HTTPS by default
- **Privacy**: No data logging or storage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web framework
- **LangChain**: For AI agent capabilities
- **OpenRouter**: For providing access to multiple AI models
- **Render**: For free hosting platform

## 📞 Support

For support, please:
- Open an issue on GitHub
- Contact: [your-email@example.com]
- LinkedIn: [your-profile]

---

**Made with ❤️ using Streamlit and AI**

> 🚀 **Deploy your own AI Data Analyst in 5 minutes!**
