from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator

from langchain.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain.vectorstores import FAISS

#from langchain.chains.question_answering import load_qa_chain
#from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

from bs4 import BeautifulSoup


Watsonx_API = ""
Project_id= ""


def get_llm():
    params = {
            GenParams.MAX_NEW_TOKENS: 256, # The maximum number of tokens that the model can generate in a single run.
            GenParams.TEMPERATURE: 0.0,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
        }
    credentials = {
            'url': "https://us-south.ml.cloud.ibm.com",
            'apikey' : Watsonx_API
        }
        
    LLAMA2_model = Model(
            model_id= 'meta-llama/llama-2-70b-chat', 
            credentials=credentials,
            params=params,
            project_id=Project_id)

    llm = WatsonxLLM(model=LLAMA2_model)
    return llm
    

def process_data():

    loader = PyPDFDirectoryLoader("info")
    docs = loader.load()
    print("number of all pages: ",len(docs))  # length of all pages together


    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 250, chunk_overlap = 50)
    
    texts = text_splitter.split_documents(docs)    

    print("number of chuncks: ",len(texts))

    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(texts, embeddings)

    return db





##---------------- Form with Label and ID ---------------###


def get_form_field_descriptions(html_file_path):
    with open(html_file_path, 'r') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all form fields (input, select, textarea)
    form_fields = soup.find_all(['input', 'select', 'textarea'])

    field_info = []
    for field in form_fields:
        field_data = {}

        # Get label text or use placeholder/name as fallback
        label = soup.find('label', {'for': field.get('id')})
        if label:
            field_data['label'] = label.get_text().strip().rstrip(':')
        else:
            placeholder = field.get('placeholder')
            name = field.get('name')
            description = placeholder if placeholder else name
            if description:
                field_data['label'] = description.strip()

        # Include the ID or name of the field
        field_id = field.get('id') or field.get('name')
        if field_id:
            field_data['id'] = field_id

        # Add the field data only if it has both an ID and a label
        if 'label' in field_data and 'id' in field_data:
            field_info.append(field_data)

    return field_info




##---------------- Response to Label and ID ---------------###

def filling_form(form_fields_info):
    llm = get_llm()
    db = process_data()

    structured_responses = []
    for field in form_fields_info:
        # Construct a directive prompt for each field
        prompt = f"Based on the document, what is the '{field['label']}'? Provide only the required information for the field ID '{field['id']}'."

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={'k': 4}),
            condense_question_prompt = PromptTemplate(input_variables=[], template=prompt),
        )

        # Call the conversation chain with the query
        result = conversation_chain({"question": prompt, "chat_history": []})

        # Add the response to the structured_responses list
        structured_responses.append({**field, "response": result['answer'].strip()})

    return structured_responses


##---------------- Running server ---------------###

from flask import Flask, jsonify
from flask_cors import CORS  # You need to install flask-cors

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/get_tax_form_data', methods=['GET'])
def get_tax_form_data():
    # Get the descriptions of form fields
    data_from_form = get_form_field_descriptions("templates/styled_tax_form.html")
    structured_responses = filling_form(data_from_form)
    
    # Convert the list of dictionaries to a single dictionary
    response_data = {field['id']: field['response'] for field in structured_responses}

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True,port=5055)
