# filematcher

PDF text extraction and semantic search using vector embeddings.

## Features

- Extract text from PDF files in a directory
- Generate embeddings using sentence-transformers
- Store embeddings in ChromaDB vector database
- Search for similar documents using semantic queries
- Automatic duplicate detection

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`:
```
TEST_DIRECTORY=/path/to/pdf/directory
```

## Usage

Run the extraction and search:
```bash
python main.py
```

This will:
1. Extract text from all PDFs in `TEST_DIRECTORY`
2. Generate embeddings for each document
3. Store embeddings in `./chroma_db/`
4. Perform a sample similarity search

## Search

Modify the query in `execute_load()` to search for specific content:
```python
query_embedding = model.encode("your search query").tolist()
```

Results show matching PDF filenames ranked by similarity.
