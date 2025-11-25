from dotenv import load_dotenv
import PyPDF2
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
import numpy as np
import chromadb

load_dotenv()

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def execute_load(pdf_texts, embeddings):
    # Creates local database in ./chroma_db folder
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="pdf_embeddings")
    
    # Get existing IDs in the database
    existing_items = collection.get()
    existing_ids = set(existing_items['ids']) if existing_items['ids'] else set()
    
    # Filter out files that already exist
    new_ids = []
    new_embeddings = []
    new_documents = []
    new_metadatas = []
    
    for filename in embeddings.keys():
        if filename not in existing_ids:
            new_ids.append(filename)
            new_embeddings.append(embeddings[filename].tolist())
            new_documents.append(pdf_texts[filename])
            new_metadatas.append({"filename": filename})
        else:
            print(f"Skipping {filename} (already exists in database)")
    
    # Add only new embeddings to ChromaDB
    if new_ids:
        collection.add(
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_documents,
            metadatas=new_metadatas
        )
        print(f"Added {len(new_ids)} new embeddings to database")
    else:
        print("No new embeddings to add")
    # Create query embedding
    query_embedding = model.encode("CPB Software (Germany) GmbH - Im Bruch 3 - 63897 Miltenberg/Main").tolist()
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # Top 3 matches
    )
    print('found ids:',results['ids'])  # Matching filenames
    print('found distances:',results['distances'])  # Similarity scores   

def execute_extraction():
    test_directory = os.environ.get('TEST_DIRECTORY')
    texts = extract_all_pdfs_from_directory(test_directory)
    print(f"Extracted texts from PDFs: {list(texts.keys())}")
    
    # Create embeddings for each PDF
    embeddings = create_embeddings(texts)
    print(f"Created embeddings for {len(embeddings)} documents")
    
    return texts, embeddings

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None


def extract_all_pdfs_from_directory(directory_path):
    """
    Extract text from all PDF files in a directory.
    
    Args:
        directory_path: Path to directory containing PDFs
        
    Returns:
        dict: {filename: extracted_text}
    """
    pdf_texts = {}
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory {directory_path} does not exist")
        return pdf_texts
    
    pdf_files = list(directory.glob('*.pdf'))
    
    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return pdf_texts
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        if text:
            pdf_texts[pdf_file.name] = text
    
    return pdf_texts


def create_embeddings(pdf_texts):
    """
    Create embeddings for each PDF text.
    
    Args:
        pdf_texts: dict of {filename: text}
        
    Returns:
        dict: {filename: embedding_vector}
    """
    embeddings = {}
    
    for filename, text in pdf_texts.items():
        # Create embedding (returns numpy array)
        embedding = model.encode(text, show_progress_bar=False)
        embeddings[filename] = embedding
        print(f"Created embedding for {filename}: shape {embedding.shape}")
    
    return embeddings


def save_embeddings(embeddings, output_dir='./embeddings'):
    """Save embeddings to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for filename, embedding in embeddings.items():
        # Save as .npy file
        output_file = output_path / f"{filename}.npy"
        np.save(output_file, embedding)
        print(f"Saved embedding to {output_file}")


if __name__ == "__main__":
    texts, embeddings = execute_extraction()
    save_embeddings(embeddings)
    execute_load(texts, embeddings)