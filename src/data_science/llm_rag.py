import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Any
import logging
from database.operations import DatabaseOperations
import json

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.db_ops = DatabaseOperations()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunk_data = []
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def create_vector_store(self, source_ids: List[int] = None):
        """Create vector store from data chunks"""
        try:
            # Get data chunks from database
            chunks_data = self.db_ops.get_data_chunks(source_ids)
            self.chunk_data = []
            
            texts = []
            for chunk in chunks_data:
                if chunk['chunk_text']:
                    texts.append(chunk['chunk_text'])
                    self.chunk_data.append(chunk)
            
            # Create embeddings
            if texts:
                embeddings = self.embedding_model.encode(texts)
                
                # Create FAISS index
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(embeddings.astype('float32'))
                
                logger.info(f"Created vector store with {len(texts)} chunks")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        if not self.index:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunk_data):
                results.append({
                    'chunk': self.chunk_data[idx],
                    'distance': distances[0][i]
                })
        
        return results
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate response using Gemini AI with RAG"""
        try:
            # Prepare context
            context = "\n\n".join([chunk['chunk']['chunk_text'] for chunk in context_chunks])
            
            prompt = f"""
            Based on the following context, please answer the question.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"
    
    def query_rag(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Complete RAG pipeline: search + generate"""
        similar_chunks = self.search_similar_chunks(query, k)
        response = self.generate_response(query, similar_chunks)
        
        return {
            'query': query,
            'response': response,
            'source_chunks': similar_chunks,
            'chunk_count': len(similar_chunks)
        }