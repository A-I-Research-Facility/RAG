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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------
# Loading data
# ---------------------------------


# read text file
def read_text_file(file_path: str):
    print("Reading a text file")
    with open(file_path, 'r', encoding="utf-8") as file:
        return file.read()


# read PDF file
def read_pdf_file(file_path: str):
    print("Reading a pdf file")
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text


# read docx file
def read_docx_file(file_path: str):
    print("Reading a docx file")
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


# read documents
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


# text = read_document("./test.pdf")
# print(text)


# ---------------------------------
# Split data
# ---------------------------------


# Spilt text data into chunks of 500 chars
def split_text(text: str, chunk_size: int = 500):
    # remove new line chars and split into sentences
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

        # if current chunk is exceeding size limit,
        # we add it to chunks and start a new current_chunk
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size

    # get the last remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


# chunks = split_text(text)
# print(len(chunks))


# ---------------------------------
# Vector embedding and storing
# ---------------------------------


# initialize chromadb client
db_client = chromadb.PersistentClient(path="./chroma_db")

# use sentence transformer embeddings
sentence_transformer_eb = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2")

collection = db_client.get_or_create_collection(
    name="documents_collection", embedding_function=sentence_transformer_eb)


# prepare data for ChromaDB
def process_document(file_path: str):
    try:
        content = read_document(file_path)
        chunks = split_text(content)

        # metadata
        file_name = os.path.basename(file_path)
        metadatas = [{"source": file_name, "chunk": i}
                     for i in range(len(chunks))]
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

        return ids, chunks, metadatas

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []


# store data in chromadb
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


# process a directory of documents
def process_and_add_documents(collection, folder_path: str):
    files = [os.path.join(folder_path, file) for file in os.listdir(
        folder_path) if os.path.isfile(os.path.join(folder_path, file))]

    for file_path in files:
        print(f"processing {os.path.basename(file_path)}...")
        ids, texts, metadatas = process_document(file_path)
        add_to_collection(collection, ids, texts, metadatas)
        print(f"added {len(texts)} chunks to collection")


process_and_add_documents(collection, "./docs")


# ---------------------------------
# Semantic search on ChromaDB
# ---------------------------------


def semantic_search(collection, query: str, n_results: int = 2):
    return collection.query(query_texts=[query], n_results=n_results)


# query = "When was GreenGrow founded?"
# results = semantic_search(collection, query)


# display results properly
def print_search_results(results):
    print("\nSearch Results:\n" + "-" * 50)

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        print(
            f"\nResult {i + 1}: Source: {meta['source']}, Chunk {meta['chunk']}")
        print(f"Content: {doc}\n")


# print_search_results(results)


# insert this string context into prompt
def get_context_with_sources(results):
    # combine chunks into single context
    context = "\n\n".join(results['documents'][0])

    # format sources with metadata
    sources = [
        f"{meta['source']} (chunk {meta['chunk']})" for meta in results['metadatas'][0]]

    return context, sources


# context, sources = get_context_with_sources(results)
# print(context)


# ---------------------------------
# Add the LLM (OpenAI)
# ---------------------------------

load_dotenv()
openrouterDeepseekKey = os.environ.get('OPENROUTER_DEEPSEEK_API_KEY')

ai_client = OpenAI(api_key=openrouterDeepseekKey,
                   base_url="https://openrouter.ai/api/v1")


def get_prompt(query: str, context: str):
    prompt = f"""Based on the following context, please answer the question.
    If answer cannot be derived from the context, 
    say 'I cannot answer this question based on the provided documents.'
        Context: {context}
        Question: {query}
        Answer:"""

    return prompt


def generate_response(query: str, context: str):
    prompt = get_prompt(query, context)

    completion = ai_client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[{
            "role": "system", "content": "You are a helpful assistant that answers questions based on given context"
        },
            {
            "role": "user",
            "content": prompt
        }],
        temperature=0,
        max_tokens=500
    )
    return completion.choices[0].message.content


# Perform RAG query
def rag_query(collection, query: str, n_chunks: int = 3):
    # get relevant chunks
    results = semantic_search(collection, query, n_chunks)
    context, sources = get_context_with_sources(results)

    # generate response
    response = generate_response(query, context)
    return response, sources


# query = "When was GreenGrow Innovations founded?"
# response, sources = rag_query(collection, query)
#
# print("\nQuery: ", query)
# print("\nAnswer: ", response)
# print("\nSources used: ")
# for source in sources:
#     print(f"- {source}")


# ---------------------------------
# Conversational RAG with memory
# ---------------------------------

# in-memory conversation store
conversations = {}


def create_session():
    session_id = str(uuid.uuid4())
    conversations[session_id] = []
    return session_id


def add_message(session_id: str, role: str, content: str):
    # create new session if new session ID
    if session_id not in conversations:
        conversations[session_id] = []

    conversations[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
