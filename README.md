# EDA Assistant

A voice-controlled desktop application for performing Exploratory Data Analysis (EDA) using natural language commands.

## Features

- Upload and analyze CSV datasets
- Voice command recognition using OpenAI Whisper
- Natural language processing of analysis commands
- Interactive visualizations and statistical analysis
- Clean, minimal GUI built with Streamlit

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Launch the application
2. Upload your CSV dataset using the file upload widget
3. Click the "Start Listening" button or press the spacebar to begin voice recognition
4. Speak your analysis commands naturally
5. View the results in the main panel

## Example Commands

- "Show summary statistics"
- "Plot a histogram of age"
- "Create a boxplot for salary"
- "Display the correlation matrix"
- "Show missing value heatmap"
- "Run linear regression with price as target and area, rooms as features"

## Requirements

- Python 3.10 or higher
- Microphone for voice input
- Speakers for optional text-to-speech output

## Note

The application runs entirely locally, with no cloud dependencies. Voice recognition is handled by OpenAI's Whisper model running on your machine. 
