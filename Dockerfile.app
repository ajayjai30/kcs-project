# Use a slightly heavier image to support sentence-transformers dependencies
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for vector operations and SQLite extensions
RUN apt-get update && apt-get install -y gcc g++ sqlite3 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your core application files
COPY improved_and_optimized_RAG.py .
COPY micro_rag_memory.py .

# Expose the FastAPI port we set in the script
EXPOSE 7860

# Start the application
CMD ["python", "improved_and_optimized_RAG.py"]