# PDF Quiz Generator with Gemini AI

This application generates quiz questions from PDF documents using Google's Gemini AI.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Edit the `.env` file and add your Gemini API key:

```env
# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-1.5-flash
```

### 3. Get Gemini API Key

1. Go to https://aistudio.google.com/app/apikey
2. Create an account or log in with your Google account
3. Click "Create API Key"
4. Copy and paste it into the `.env` file

### 4. Run the Application

```bash
streamlit run app.py
```

## Features

-   **Robust PDF Extraction**: Handles text, images, tables, and special characters
-   **AI-Powered Question Generation**: Uses Google Gemini AI models
-   **Multiple Difficulty Levels**: Easy, Medium, Hard questions
-   **Question Types**: Multiple choice and True/False
-   **Export Options**: JSON and text formats
-   **Fallback Mode**: Works without API keys using simulated generation

## Usage

1. Upload a PDF file (max 200MB)
2. Choose difficulty level and number of questions
3. Click "Generate Quiz from PDF"
4. Review and export your questions

## Supported PDF Content

-   Text content (with special characters and Unicode)
-   Images (extracted but not displayed)
-   Tables and graphs (text extracted)
-   Mixed content documents

## Troubleshooting

If you encounter issues:

1. Check that your Gemini API key is valid
2. Ensure the PDF contains readable text
3. Try a different PDF if extraction fails
4. Use fallback mode if API limits are reached
