from dotenv import load_dotenv
import PyPDF2
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
import numpy as np

load_dotenv()

# Load the embedding model (only ~80MB, no API key needed)
model = SentenceTransformer('all-MiniLM-L6-v2')

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