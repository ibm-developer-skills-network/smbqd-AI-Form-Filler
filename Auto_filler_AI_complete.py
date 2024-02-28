# Import required libraries from various modules.
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

# langchain library for embeddings, text splitting, and conversational retrieval.
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate

# Document loader and vector store modules for processing PDFs.
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS

# BeautifulSoup for parsing HTML content.
from bs4 import BeautifulSoup

# Flask for web server and CORS for cross-origin resource sharing.
from flask import Flask, jsonify, render_template
from flask_cors import CORS

#------------> Function to initialize and retrieve the language model from IBM Watson. <------------
def get_llm():
    # Credentials for accessing IBM Watson services.
    my_credentials = {
        "url": "https://us-south.ml.cloud.ibm.com"
    }

    # Parameters for controlling the generation of text by the model.
    params = {
        GenParams.MAX_NEW_TOKENS: 256, # Maximum tokens generated per run.
        GenParams.TEMPERATURE: 0.0,    # Controls randomness in generation.
    }

    # Initialize the model with specified ID and credentials.
    LLAMA2_model = Model(
        model_id='meta-llama/llama-2-70b-chat', 
        credentials=my_credentials,
        params=params,
        project_id="skills-network")

    # Create and return the Watson Language Model.
    llm = WatsonxLLM(model=LLAMA2_model)
    return llm

#------------> Function to process and index PDF documents. <------------
def process_data():
    # Load PDF documents from a specified directory.
    loader = PyPDFDirectoryLoader("info")
    docs = loader.load()

    # Split the text from documents for better processing.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)

    # Create embeddings for the text data.
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Index the documents using FAISS for efficient retrieval.
    db = FAISS.from_documents(texts, embeddings)
    return db

#------------> Function to extract form field descriptions from an HTML file. <------------
def get_form_field_descriptions(html_file_path):
    with open(html_file_path, 'r') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Find and process all form fields in the HTML.
    form_fields = soup.find_all(['input', 'select', 'textarea'])
    field_info = []
    for field in form_fields:
        field_data = {}

        # Extract label text or use placeholder/name as a fallback.
        label = soup.find('label', {'for': field.get('id')})
        if label:
            field_data['label'] = label.get_text().strip().rstrip(':')
        else:
            placeholder = field.get('placeholder')
            name = field.get('name')
            description = placeholder if placeholder else name
            if description:
                field_data['label'] = description.strip()

        # Include the ID or name of the field in the data.
        field_id = field.get('id') or field.get('name')
        if field_id:
            field_data['id'] = field_id

        # Add complete field data to the list.
        if 'label' in field_data and 'id' in field_data:
            field_info.append(field_data)

    return field_info

#------------> Function to automate form filling using the processed data. <------------
def filling_form(form_fields_info):
    # Initialize the language model and data processing tools.
    llm = get_llm()
    db = process_data()

    structured_responses = []
    for field in form_fields_info:
        # Create a specific prompt for each form field.
        prompt = f"Based on the document, what is the '{field['label']}'? Provide only the required information for the field ID '{field['id']}'."

        # Set up a conversational chain to retrieve and generate responses.
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={'k': 4}),
            condense_question_prompt=PromptTemplate(input_variables=[], template=prompt),
        )

        # Get the response for each field.
        result = conversation_chain({"question": prompt, "chat_history": []})
        structured_responses.append({**field, "response": result['answer'].strip()})

    return structured_responses

#------------> Initialize the Flask application for the web server. <------------
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests.

# Define route for the home page.
@app.route('/')
def home():
    return render_template('styled_tax_form.html')

# Define API route to retrieve form data.
@app.route('/api/get_tax_form_data', methods=['GET'])
def get_tax_form_data():
    data_from_form = get_form_field_descriptions("templates/styled_tax_form.html")
    structured_responses = filling_form(data_from_form)

    # Convert responses to a JSON format for the frontend.
    response_data = {field['id']: field['response'] for field in structured_responses}
    return jsonify(response_data)

# Run the Flask application if this script is executed directly.
if __name__ == '__main__':
    app.run(debug=True, port=5055)
