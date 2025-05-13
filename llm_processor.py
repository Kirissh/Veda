import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import logging
import google.generativeai as genai
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
GEMINI_API_KEY = "AIzaSyAyCdk61jjV8Hn2_1Uq4434rJs7d_fHS_c"
genai.configure(api_key=GEMINI_API_KEY)

class LLMProcessor:
    def __init__(self):
        """Initialize the LLM processor with Gemini 2.0 Flash"""
        try:
            # Configure Gemini
            genai.configure(api_key="AIzaSyDxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXx")  # Replace with your API key
            self.model = genai.GenerativeModel('gemini-1.0-flash')
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            self.model = None

    def generate_insights(self, dataset_summary: str) -> str:
        """Generate insights about the dataset using Gemini"""
        try:
            if not self.model:
                return "LLM model not available. Please check your API key."

            prompt = f"""
            Analyze this dataset summary and provide key insights:
            {dataset_summary}
            
            Focus on:
            1. Data quality issues
            2. Potential relationships between columns
            3. Interesting patterns or anomalies
            4. Suggestions for further analysis
            
            Keep the response concise and actionable.
            """

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return f"Error generating insights: {str(e)}"

    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the dataset and return insights"""
        try:
            # Basic dataset analysis
            analysis = {
                "data_quality": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "duplicate_rows": df.duplicated().sum(),
                    "missing_values": df.isnull().sum().to_dict()
                },
                "column_types": {},
                "insights": []
            }
            
            # Analyze each column
            for col in df.columns:
                col_type = str(df[col].dtype)
                unique_vals = df[col].nunique()
                null_count = df[col].isnull().sum()
                null_percent = (null_count / len(df)) * 100
                
                analysis["column_types"][col] = {
                    "type": col_type,
                    "unique_values": unique_vals,
                    "null_percentage": null_percent
                }
            
            # Generate insights using Gemini
            dataset_summary = f"""
            Dataset Summary:
            - Total Rows: {len(df)}
            - Total Columns: {len(df.columns)}
            - Column Types: {df.dtypes.to_dict()}
            - Missing Values: {df.isnull().sum().to_dict()}
            - Sample Data: {df.head().to_dict()}
            """
            analysis["insights"] = self.generate_insights(dataset_summary)
            
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing dataset: {str(e)}")
            return {
                "error": str(e),
                "data_quality": {},
                "column_types": {},
                "insights": []
            }

    def get_suggestions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get AI-powered suggestions for analysis"""
        try:
            if not self.model:
                return self._get_default_suggestions(df)

            # Prepare dataset summary for Gemini
            dataset_summary = f"""
            Dataset Summary:
            - Total Rows: {len(df)}
            - Total Columns: {len(df.columns)}
            - Column Types: {df.dtypes.to_dict()}
            - Missing Values: {df.isnull().sum().to_dict()}
            - Sample Data: {df.head().to_dict()}
            """

            prompt = f"""
            Based on this dataset summary, suggest relevant analysis commands:
            {dataset_summary}
            
            Return the suggestions in this JSON format:
            {{
                "categories": [
                    {{
                        "category": "string",
                        "commands": [
                            {{
                                "text": "string",
                                "command": "string"
                            }}
                        ]
                    }}
                ]
            }}
            
            Focus on:
            1. Data quality and preprocessing
            2. Basic statistical analysis
            3. Visualizations
            4. Advanced analysis
            """

            response = self.model.generate_content(prompt)
            suggestions = json.loads(response.text)
            return suggestions.get("categories", [])

        except Exception as e:
            logger.error(f"Error getting suggestions: {str(e)}")
            return self._get_default_suggestions(df)

    def _get_default_suggestions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get default suggestions when LLM is not available"""
        return [
            {
                "category": "Data Preprocessing",
                "commands": [
                    {"text": "Show Missing Values", "command": "show missing values"},
                    {"text": "Remove Duplicates", "command": "remove duplicates"}
                ]
            },
            {
                "category": "Basic Analysis",
                "commands": [
                    {"text": "Show Summary Statistics", "command": "show summary statistics"},
                    {"text": "Show Top 5 Rows", "command": "show top 5 rows"}
                ]
            },
            {
                "category": "Visualizations",
                "commands": [
                    {"text": "Show Correlation Matrix", "command": "show correlation matrix"},
                    {"text": "Plot Histogram", "command": f"plot histogram of {df.columns[0]}"}
                ]
            }
        ]

    def process_command(self, command: str, available_columns: List[str]) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Process a command using Gemini"""
        try:
            if not self.model:
                return None, False

            prompt = f"""
            Interpret this data analysis command and return the analysis type and parameters:
            Command: {command}
            Available Columns: {available_columns}
            
            Return the response in this JSON format:
            {{
                "command_type": "string",
                "parameters": {{
                    "column": "string",
                    "target": "string",
                    "features": "string"
                }},
                "explanation": "string"
            }}
            
            Valid command types:
            - summary
            - histogram
            - boxplot
            - correlation
            - missing
            - regression
            - scatter
            """

            response = self.model.generate_content(prompt)
            result = json.loads(response.text)
            return result, True

        except Exception as e:
            logger.error(f"Error processing command: {str(e)}")
            return None, False

    def validate_command(self, command_result: Dict[str, Any], available_columns: List[str]) -> Tuple[bool, str]:
        """Validate the command result"""
        try:
            if not command_result:
                return False, "Invalid command result"

            command_type = command_result.get("command_type")
            parameters = command_result.get("parameters", {})

            if not command_type:
                return False, "Missing command type"

            # Validate column names
            for param_name, param_value in parameters.items():
                if param_name in ["column", "target", "features"]:
                    if isinstance(param_value, str):
                        if param_value not in available_columns:
                            return False, f"Column '{param_value}' not found in dataset"
                    elif isinstance(param_value, list):
                        for col in param_value:
                            if col not in available_columns:
                                return False, f"Column '{col}' not found in dataset"

            return True, "Command is valid"

        except Exception as e:
            logger.error(f"Error validating command: {str(e)}")
            return False, f"Error validating command: {str(e)}" 