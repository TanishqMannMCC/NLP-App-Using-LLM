Gemini 2.5 Flash NLP Agent

This is a comprehensive web application built with Streamlit and the Gemini 2.5 Flash model. It serves as a unified platform for a wide range of Natural Language Processing (NLP) tasks, demonstrating the power and versatility of a single Large Language Model through prompt engineering.

Instead of relying on multiple separate models for each task, this application uses a single LLM as a central, multi-talented agent.

Features

The app contains a sidebar that allows you to perform the following NLP tasks:

● Sentiment & Emotion Analysis: Determines the emotional tone and specific emotions within a text.

● Text Summarization: Condenses long documents into concise, abstractive summaries.

● Text Classification: Categorizes a given text into one of a list of predefined labels.

● Machine Translation: Translates text from one language to another.

● Question Answering: Answers questions based on a provided document, without using external knowledge.

● Text Generation: Generates creative and coherent text based on a user-provided prompt.

● Topic Modeling & Keyword Extraction: Identifies the main topics and relevant keywords in a document.

● Grammatical Error Correction: Corrects grammatical mistakes in a given text.

● Chatbot: A conversational agent that maintains context and responds in a natural, non-technical tone.

Setup and Installation

1. Clone the Repository (or create the files): First, ensure you have a local copy of the project files (app.py and requirements.txt).

2. Create a Python Virtual Environment: (Optional, but recommended) python -m venv venv source venv/bin/activate # On macOS/Linux venv\Scripts\activate # On Windows

3. Install Dependencies: Install the necessary libraries using the requirements.txt file. pip install -r requirements.txt

4. Set Your Gemini API Key: You must obtain a valid API key from Google AI Studio. Do not put this key directly in your code. Instead, set it as an environment variable in your terminal.

○ For macOS/Linux: export GEMINI_API_KEY="YOUR_API_KEY_HERE"

○ For Windows (Command Prompt): set GEMINI_API_KEY=YOUR_API_KEY_HERE

○ For Windows (PowerShell): $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"

How to Run the App

After completing the setup, run the following command in the same terminal session where you set your API key:

streamlit run app.py

The application will launch in your web browser, and you can begin using all the available NLP tasks.
