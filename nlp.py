import streamlit as st
import google.generativeai as genai
import json
import os
import docx
import fitz # PyMuPDF
import sys # Import the sys module to handle exceptions

# Set up the Gemini API key.
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    st.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()
else:
    genai.configure(api_key=API_KEY)

# Define the model to be used
MODEL_NAME = "gemini-2.5-flash-preview-05-20"

# Main Streamlit app title and description
st.set_page_config(
    page_title="Gemini 2.5 Flash NLP Agent",
    layout="wide",
)

st.title("The Gemini 2.5 Flash NLP Agent ⚡")
st.markdown("This application demonstrates the power of the Gemini 2.5 Flash model for a comprehensive range of NLP tasks.")

# Use a sidebar for task selection
st.sidebar.header("Select an NLP Task")
task = st.sidebar.selectbox(
    "Choose a task from the list below:",
    [
        "Sentiment & Emotion Analysis",
        "Text Summarization",
        "Text Classification",
        "Machine Translation",
        "Question Answering",
        "Text Generation",
        "Topic Modeling & Keyword Extraction",
        "Grammatical Error Correction",
        "Chatbot"
    ]
)

st.sidebar.markdown("""
---
**How it Works:**
This app uses a single LLM (`gemini-2.5-flash`)
and custom prompts to perform all these tasks.
No separate models are needed for each functionality.
""")

def read_file_content(file):
    """Reads the content of various file types and returns it as a single string."""
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'txt':
        return file.read().decode('utf-8')
    elif file_extension == 'docx':
        try:
            doc = docx.Document(file)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
        except Exception as e:
            st.error(f"Error reading DOCX file: {e}")
            return None
    elif file_extension == 'pdf':
        try:
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return None
    else:
        st.warning(f"Unsupported file type: {file_extension}. Please upload a .txt, .docx, or .pdf file.")
        return None

# --- Task 1: Sentiment & Emotion Analysis ---
if task == "Sentiment & Emotion Analysis":
    st.header("Sentiment & Emotion Analysis")
    st.markdown("Determines the emotional tone and sentiment of a text.")
    
    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="e.g., I'm so thrilled with this product! It's fantastic.",
        height=150
    )

    if st.button("Analyze Sentiment", key="sentiment"):
        if text_input:
            with st.spinner("Analyzing text..."):
                prompt = f"""
                Analyze the following text and provide a JSON object with:
                1. The overall sentiment (e.g., 'Positive', 'Negative', 'Neutral').
                2. A primary emotion (e.g., 'joy', 'anger', 'sadness', 'surprise').
                3. A brief, one-sentence justification.
                
                Text: '{text_input}'
                """
                try:
                    response = genai.GenerativeModel(MODEL_NAME).generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            response_mime_type="application/json",
                            temperature=0.1
                        )
                    )
                    st.subheader("Results:")
                    st.json(json.loads(response.text))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to analyze.")

# --- Task 2: Text Summarization ---
elif task == "Text Summarization":
    st.header("Text Summarization")
    st.markdown("Condenses a lengthy document into a concise summary.")
    
    doc_input = st.text_area(
        "Enter a long document or article:",
        height=300,
        placeholder="Paste a long piece of text here."
    )

    if st.button("Generate Summary", key="summarize"):
        if doc_input:
            with st.spinner("Generating summary..."):
                prompt = f"""
                Provide a concise, abstractive summary of the following document.
                The summary should be no more than 150 words and should cover all key points.
                
                Document: '{doc_input}'
                """
                try:
                    response = genai.GenerativeModel(MODEL_NAME).generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.5
                        )
                    )
                    st.subheader("Summary:")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to summarize.")

# --- Task 3: Text Classification ---
elif task == "Text Classification":
    st.header("Text Classification")
    st.markdown("Categorizes text into one of the predefined labels.")
    
    text_to_classify = st.text_area(
        "Enter text to classify:",
        height=150,
        placeholder="e.g., The stock market fell today after a surprising earnings report."
    )
    
    labels = st.text_input(
        "Enter possible labels, separated by commas:",
        value="Technology, Finance, Sports, Politics, Health"
    )

    if st.button("Classify Text", key="classify"):
        if text_to_classify and labels:
            with st.spinner("Classifying..."):
                label_list = [l.strip() for l in labels.split(',')]
                prompt = f"""
                Classify the following text into one of these categories: {', '.join(label_list)}.
                Provide only the category name as the output, with no extra text.
                
                Text: '{text_to_classify}'
                """
                try:
                    response = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
                    st.subheader("Predicted Category:")
                    st.success(response.text.strip())
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please provide text and a list of labels.")

# --- Task 4: Machine Translation ---
elif task == "Machine Translation":
    st.header("Machine Translation")
    st.markdown("Translates text from one language to another.")
    
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox("Source Language:", ["English", "Spanish", "French", "German"])
    with col2:
        target_lang = st.selectbox("Target Language:", ["French", "English", "Spanish", "German"])

    text_to_translate = st.text_area(
        f"Enter text in {source_lang}:",
        height=150,
        placeholder=f"e.g., Hello, how are you?"
    )

    if st.button("Translate", key="translate"):
        if text_to_translate:
            with st.spinner("Translating..."):
                prompt = f"Translate the following {source_lang} text to {target_lang}: '{text_to_translate}'"
                try:
                    response = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
                    st.subheader("Translated Text:")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to translate.")

# --- Task 5: Question Answering ---
elif task == "Question Answering":
    st.header("Question Answering")
    st.markdown("Answers questions based on a provided document.")

    document = st.text_area(
        "Enter a document (context):",
        height=200,
        placeholder="e.g., The Amazon rainforest is a moist broadleaf forest in the Amazon biome that covers most of the Amazon basin of South America."
    )
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the Amazon rainforest?"
    )

    if st.button("Get Answer", key="qa"):
        if document and question:
            with st.spinner("Finding answer..."):
                prompt = f"""
                Use ONLY the following document to answer the question.
                Do not use any outside knowledge.
                
                Document: '{document}'
                
                Question: {question}
                """
                try:
                    response = genai.GenerativeModel(MODEL_NAME).generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.2
                        )
                    )
                    st.subheader("Answer:")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please provide a document and a question.")

# --- Task 6: Text Generation ---
elif task == "Text Generation":
    st.header("Text Generation")
    st.markdown("Generates creative text based on your prompt.")
    
    generation_prompt = st.text_area(
        "Give the LLM a prompt:",
        height=150,
        placeholder="e.g., Write a short, humorous story about a robot trying to bake a cake."
    )

    if st.button("Generate Text", key="generate"):
        if generation_prompt:
            with st.spinner("Generating..."):
                try:
                    response = genai.GenerativeModel(MODEL_NAME).generate_content(generation_prompt)
                    st.subheader("Generated Text:")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a prompt for text generation.")

# --- Task 7: Topic Modeling & Keyword Extraction ---
elif task == "Topic Modeling & Keyword Extraction":
    st.header("Topic Modeling & Keyword Extraction")
    st.markdown("Identifies the main topics and key keywords from a text.")
    
    text_to_analyze = st.text_area(
        "Enter a document to analyze:",
        height=300,
        placeholder="Paste a document here."
    )

    if st.button("Extract Topics and Keywords", key="topics_keywords"):
        if text_to_analyze:
            with st.spinner("Analyzing text..."):
                prompt = f"""
                Analyze the following document and provide a JSON object with:
                1. A list of the main topics.
                2. A list of the most relevant keywords.
                
                Document: '{text_to_analyze}'
                """
                try:
                    response = genai.GenerativeModel(MODEL_NAME).generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            response_mime_type="application/json",
                            temperature=0.1
                        )
                    )
                    st.subheader("Results:")
                    st.json(json.loads(response.text))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a document to analyze.")

# --- Task 8: Grammatical Error Correction ---
elif task == "Grammatical Error Correction":
    st.header("Grammatical Error Correction")
    st.markdown("Identifies and corrects grammatical errors in text.")
    
    text_with_errors = st.text_area(
        "Enter text to correct:",
        height=150,
        placeholder="e.g., He don't have no money for buy a car."
    )

    if st.button("Correct Grammar", key="correct"):
        if text_with_errors:
            with st.spinner("Correcting..."):
                prompt = f"Correct the grammatical errors in the following sentence: '{text_with_errors}'"
                try:
                    response = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
                    st.subheader("Corrected Text:")
                    st.success(response.text)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to correct.")

# --- Task 9: Chatbot ---
elif task == "Chatbot":
    st.header("Chat with your Document")
    st.markdown("Upload a document and ask questions about its content. The bot will only use the provided document to answer. If no document is uploaded, it will be a general-purpose chatbot.")

    # New file uploader widget
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, .docx, or .pdf only)",
        type=["txt", "docx", "pdf"]
    )
    
    # Store the document content in session state
    if uploaded_file:
        if "document_content" not in st.session_state or st.session_state.document_name != uploaded_file.name:
            try:
                doc_content = read_file_content(uploaded_file)
                st.session_state.document_content = doc_content
                st.session_state.document_name = uploaded_file.name
                if doc_content:
                    st.success(f"Successfully loaded '{uploaded_file.name}'! You can now ask questions.")
                st.session_state.messages = [] # Clear history for new document
            except Exception as e:
                st.error(f"Error processing file: {e}")
                if "document_content" in st.session_state:
                    del st.session_state.document_content
                if "document_name" in st.session_state:
                    del st.session_state.document_name
    else:
        # If no file is uploaded, clear document-related state
        if "document_content" in st.session_state:
            del st.session_state.document_content
        if "document_name" in st.session_state:
            del st.session_state.document_name

    # Set system instruction dynamically based on whether a document is loaded
    if "document_content" in st.session_state and st.session_state.document_content:
        system_instruction_text = (
            "You are a helpful assistant specialized in answering questions about the provided document. "
            "Your responses should be in plain text and based ONLY on the document content. "
            "Do not use any outside knowledge. If the answer is not in the document, say 'I cannot find the answer in the provided document.' Then, if the topic is well-known, offer to explain it from your general knowledge. For example, 'However, I can provide a general explanation of [topic] if you would like.' Respond to the user with this format."
        )
    else:
        system_instruction_text = (
            "You are a general-purpose conversational chatbot. Your responses should be in plain text and should avoid using code blocks unless the user explicitly requests code or asks a technical question requiring code."
        )

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            {"role": "model", "content": "Hello! I am a general-purpose chatbot. How can I help you today?"}
        )

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
         
        with st.spinner("Thinking..."):
            try:
                # Construct the prompt with the document content if it exists
                if "document_content" in st.session_state and st.session_state.document_content:
                    full_prompt = (
                        f"Document content:\n{st.session_state.document_content}\n\n"
                        f"Question: {prompt}"
                    )
                else:
                    full_prompt = prompt
                
                # Use the model to generate the response, passing the system instruction
                # and a history based on session state messages
                model = genai.GenerativeModel(
                    MODEL_NAME,
                    system_instruction=system_instruction_text
                )
                
                # Start a new chat session with the full message history
                chat_session = model.start_chat(history=[
                    {"role": m["role"], "parts": [m["content"]]}
                    for m in st.session_state.messages
                ])

                # Send the new message and stream the response
                response = chat_session.send_message(full_prompt, stream=True)
                
                # Display the streaming response and build the final response text
                with st.chat_message("model"):
                    final_response_text = ""
                    for chunk in response:
                        final_response_text += chunk.text
                    st.markdown(final_response_text)
                
                # Update chat history with the assistant's final response
                st.session_state.messages.append({"role": "model", "content": final_response_text})
            except Exception as e:
                st.error(f"An error occurred: {e}")
