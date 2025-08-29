import os
import docx
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI
import uuid
from datetime import datetime
import json
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './docs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ChromaDB
db_client = chromadb.PersistentClient(path="./chroma_db")
sentence_transformer_eb = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2")
collection = db_client.get_or_create_collection(
    name="documents_collection", embedding_function=sentence_transformer_eb)

# Initialize OpenAI client
load_dotenv()
openrouterDeepseekKey = os.environ.get('OPENROUTER_DEEPSEEK_API_KEY')
ai_client = OpenAI(api_key=openrouterDeepseekKey,
                   base_url="https://openrouter.ai/api/v1")

# In-memory conversation store
conversations = {}
conversations_lock = threading.Lock()

# ---------------------------------
# Helper Functions
# ---------------------------------

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def read_text_file(file_path: str):
    with open(file_path, 'r', encoding="utf-8") as file:
        return file.read()

def read_pdf_file(file_path: str):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def read_docx_file(file_path: str):
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def read_document(file_path: str):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    switcher = {
        '.txt': read_text_file,
        '.pdf': read_pdf_file,
        '.docx': read_docx_file
    }

    func = switcher.get(file_extension)
    if func:
        return func(file_path)
    else:
        return f"Unsupported file type: {file_extension}"

def split_text(text: str, chunk_size: int = 500):
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if not sentence.endswith('.'):
            sentence += '.'

        sentence_size = len(sentence)

        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def process_document(file_path: str):
    try:
        content = read_document(file_path)
        chunks = split_text(content)

        file_name = os.path.basename(file_path)
        metadatas = [{"source": file_name, "chunk": i}
                     for i in range(len(chunks))]
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

        return ids, chunks, metadatas

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []

def add_to_collection(collection, ids, texts, metadatas):
    if not texts:
        return

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        collection.add(
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )

def semantic_search(collection, query: str, n_results: int = 2):
    return collection.query(query_texts=[query], n_results=n_results)

def get_context_with_sources(results):
    context = "\n\n".join(results['documents'][0])
    sources = [
        f"{meta['source']} (chunk {meta['chunk']})" for meta in results['metadatas'][0]]
    return context, sources

def get_prompt(query: str, conversation_history, context: str):
    prompt = f"""Based on the following context and conversation history,
    please provide a relevant and contextual response.
    If answer cannot be derived from the context, only use conversation history or 
    say 'I cannot answer this question based on the provided information.'
        Context from documents: {context}
        
        Previous conversation: {conversation_history}

        human: {query}

        assistant:"""

    return prompt

def generate_response(query: str, context: str, conversation_history: str = ""):
    prompt = get_prompt(query, conversation_history, context)

    completion = ai_client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[{
            "role": "system", "content": prompt
        }],
        temperature=0,
        max_tokens=500
    )
    return completion.choices[0].message.content

def rag_query(collection, query: str, n_chunks: int = 3):
    results = semantic_search(collection, query, n_chunks)
    context, sources = get_context_with_sources(results)
    response = generate_response(query, context)
    return response, sources

def create_session():
    session_id = str(uuid.uuid4())
    with conversations_lock:
        conversations[session_id] = []
    return session_id

def add_message(session_id: str, role: str, content: str):
    with conversations_lock:
        if session_id not in conversations:
            conversations[session_id] = []

        conversations[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

def get_conversation_history(session_id: str, max_messages: int = None):
    with conversations_lock:
        if session_id not in conversations:
            return []

        history = conversations[session_id]

        if max_messages:
            history = history[-max_messages:]

        return history

def format_history_for_prompt(session_id: str, max_messages: int = 5):
    history = get_conversation_history(session_id, max_messages)
    formatted_history = ""

    for msg in history:
        role = "human" if msg["role"] == "user" else "assistant"
        formatted_history += f"{role}: {msg['content']}\n\n"

    return formatted_history

def contextualize_query(query: str, conversation_history, ai_client):
    contextualize_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed, otherwise return it as is."
    )

    completion = ai_client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[{
            "role": "system", "content": contextualize_system_prompt
        },
        {
            "role": "user", "content": f"Chat history:\n{conversation_history}\n\nQuestion:\n{query}"
        }
        ],
    )
    return completion.choices[0].message.content

def conversational_rag_query(collection, query: str, session_id: str, n_chunks: int = 3):
    conversation_history = format_history_for_prompt(session_id)
    query = contextualize_query(query, conversation_history, ai_client)
    
    results = semantic_search(collection, query, n_chunks)
    context, sources = get_context_with_sources(results)
    response = generate_response(query, context, conversation_history)

    add_message(session_id, "user", query)
    add_message(session_id, "assistant", response)

    return response, sources

# ---------------------------------
# API Routes
# ---------------------------------

@app.route('/')
def index():
    return jsonify({"message": "Document RAG API", "status": "active"})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process and add to collection
        ids, texts, metadatas = process_document(file_path)
        add_to_collection(collection, ids, texts, metadatas)
        
        return jsonify({
            "message": "File uploaded and processed successfully",
            "filename": filename,
            "chunks_added": len(texts)
        }), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
    
    question = data['question']
    session_id = data.get('session_id')
    n_chunks = data.get('n_chunks', 3)
    
    if not session_id:
        session_id = create_session()
    
    try:
        response, sources = conversational_rag_query(collection, question, session_id, n_chunks)
        return jsonify({
            "answer": response,
            "sources": sources,
            "session_id": session_id
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/session', methods=['POST'])
def create_new_session():
    session_id = create_session()
    return jsonify({"session_id": session_id}), 200

@app.route('/session/<session_id>', methods=['GET'])
def get_session_history(session_id):
    history = get_conversation_history(session_id)
    return jsonify({"session_id": session_id, "history": history}), 200

@app.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    with conversations_lock:
        if session_id in conversations:
            del conversations[session_id]
            return jsonify({"message": "Session deleted successfully"}), 200
        else:
            return jsonify({"error": "Session not found"}), 404

@app.route('/documents', methods=['GET'])
def list_documents():
    try:
        # Get unique documents from collection metadata
        results = collection.get()
        sources = set()
        for metadata in results['metadatas']:
            sources.add(metadata['source'])
        
        return jsonify({"documents": list(sources)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Process any existing documents in the docs folder on startup
    if os.path.exists(app.config['UPLOAD_FOLDER']) and os.listdir(app.config['UPLOAD_FOLDER']):
        print("Processing existing documents...")
        files = [os.path.join(app.config['UPLOAD_FOLDER'], file) for file in os.listdir(
            app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], file))]
        
        for file_path in files:
            print(f"Processing {os.path.basename(file_path)}...")
            ids, texts, metadatas = process_document(file_path)
            add_to_collection(collection, ids, texts, metadatas)
            print(f"Added {len(texts)} chunks to collection")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
