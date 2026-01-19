# 📊 AI Data Analyst

A professional Streamlit application that allows users to upload data files (CSV, Excel, PDF, Word) and ask questions in plain English using AI.

## 🚀 Deployment Guide

### Step 1: Files Ready ✅
- `app.py` - Main application file
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Theme configuration

### Step 2: GitHub Setup
1. Open GitHub Desktop
2. Create a new repository named `ai-data-analyst`
3. Add all files and commit
4. Publish repository to GitHub

### Step 3: Render Deployment
1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Create a free account
3. Click **New +** → **Web Service**
4. Connect your GitHub repository
5. Configure settings:
   - **Name**: `my-data-analyst`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py`
6. Add Environment Variable:
   - **Key**: `OPENROUTER_API_KEY`
   - **Value**: Your OpenRouter API key
7. Click **Create Web Service**

## 🎨 Features
- 📁 Universal file support (CSV, Excel, PDF, Word)
- 💬 Modern chat interface
- 📊 Real-time data dashboard
- ⚡ Fast sampling mode
- 🎛️ Professional UI/UX
- 🛡️ Robust error handling

## 🔧 Configuration
- **Theme**: Professional blue color scheme
- **Models**: Meta Llama 3.1 8B (default), Gemini 2.0 Flash
- **Performance**: Data caching and sampling
- **Security**: API key stored in environment variables

## 📱 Usage
1. Upload your data file
2. Ask questions in plain English
3. Get instant AI-powered insights
4. Download results as needed

## 🌐 Live Demo
Once deployed, your app will be available at: `https://my-data-analyst.onrender.com`
