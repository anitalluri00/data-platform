import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import logging

# Add src to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from database.models import create_tables, SessionLocal, DataSource, RawData, ProcessedData, DataChunks, VectorStore
from database.operations import DatabaseOperations
from data_engineering.ingestion import DataIngestion
from data_engineering.etl_pipeline import ETLPipeline
from data_science.llm_rag import RAGSystem
from data_science.ml_models import MLModelManager
from utils.file_processor import FileProcessor

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize database
create_tables()

class DataPlatformApp:
    def __init__(self):
        self.db_ops = DatabaseOperations()
        self.ingestion = DataIngestion()
        self.etl_pipeline = ETLPipeline()
        self.file_processor = FileProcessor()
        
        # Initialize RAG system if API key is available
        if settings.GEMINI_API_KEY:
            self.rag_system = RAGSystem(settings.GEMINI_API_KEY)
        else:
            self.rag_system = None
        
        self.ml_manager = MLModelManager()
    
    def run(self):
        st.set_page_config(
            page_title="Data Engineering & Science Platform",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🚀 Data Engineering & Science Platform")
        st.markdown("---")
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        app_mode = st.sidebar.selectbox(
            "Choose Module",
            [
                "Data Ingestion",
                "ETL Pipeline", 
                "Data Analytics",
                "Machine Learning",
                "RAG System",
                "Admin Dashboard"
            ]
        )
        
        # Authentication for admin features
        if app_mode == "Admin Dashboard":
            self._admin_auth()
        
        # Route to selected module
        if app_mode == "Data Ingestion":
            self.data_ingestion_ui()
        elif app_mode == "ETL Pipeline":
            self.etl_pipeline_ui()
        elif app_mode == "Data Analytics":
            self.data_analytics_ui()
        elif app_mode == "Machine Learning":
            self.ml_ui()
        elif app_mode == "RAG System":
            self.rag_ui()
        elif app_mode == "Admin Dashboard":
            self.admin_dashboard_ui()
    
    def _admin_auth(self):
        """Simple admin authentication"""
        if 'admin_authenticated' not in st.session_state:
            st.session_state.admin_authenticated = False
        
        if not st.session_state.admin_authenticated:
            st.sidebar.subheader("Admin Login")
            password = st.sidebar.text_input("Admin Password", type="password")
            if st.sidebar.button("Login"):
                # In production, use proper authentication
                if password == "admin123":  # Change this in production
                    st.session_state.admin_authenticated = True
                    st.sidebar.success("Logged in successfully!")
                else:
                    st.sidebar.error("Invalid password!")
    
    def data_ingestion_ui(self):
        st.header("📥 Data Ingestion")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("File Upload")
            uploaded_files = st.file_uploader(
                "Upload Files",
                type=[
                    'pdf', 'doc', 'docx', 'txt', 'rtf', 
                    'xlsx', 'xls', 'csv', 'ppt', 'pptx',
                    'jpg', 'jpeg', 'png', 'gif', 'bmp'
                ],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    # Save file temporarily
                    file_path = f"data/raw/{uploaded_file.name}"
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        source_id = self.ingestion.ingest_file(file_path, uploaded_file.name)
                        st.success(f"✅ Successfully ingested: {uploaded_file.name} (ID: {source_id})")
                    except Exception as e:
                        st.error(f"❌ Failed to ingest {uploaded_file.name}: {str(e)}")
        
        with col2:
            st.subheader("Web Content")
            url = st.text_input("Enter URL")
            source_name = st.text_input("Source Name")
            
            if st.button("Ingest Web Content"):
                if url and source_name:
                    try:
                        source_id = self.ingestion.ingest_web_content(url, source_name)
                        st.success(f"✅ Successfully ingested web content (ID: {source_id})")
                    except Exception as e:
                        st.error(f"❌ Failed to ingest web content: {str(e)}")
                else:
                    st.warning("Please provide both URL and Source Name")
        
        # Show recent data sources
        st.subheader("Recent Data Sources")
        sources = self.db_ops.get_recent_sources(limit=10)
        if sources:
            df_sources = pd.DataFrame(sources)
            st.dataframe(df_sources)
        else:
            st.info("No data sources found")
    
    def etl_pipeline_ui(self):
        st.header("🔄 ETL Pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pipeline Control")
            if st.button("Run ETL Pipeline"):
                with st.spinner("Running ETL Pipeline..."):
                    try:
                        self.etl_pipeline.run_pipeline()
                        st.success("✅ ETL Pipeline completed successfully!")
                    except Exception as e:
                        st.error(f"❌ ETL Pipeline failed: {str(e)}")
            
            st.subheader("Data Quality")
            quality_data = self.db_ops.get_data_quality_metrics()
            if quality_data:
                df_quality = pd.DataFrame(quality_data)
                st.dataframe(df_quality)
                
                # Quality score visualization
                fig = px.bar(df_quality, x='data_type', y='avg_quality_score', 
                           title='Average Quality Score by Data Type')
                st.plotly_chart(fig)
        
        with col2:
            st.subheader("Processed Data Overview")
            stats = self.db_ops.get_processing_stats()
            if stats:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Sources", stats['total_sources'])
                col2.metric("Processed Records", stats['processed_records'])
                col3.metric("Avg Quality Score", f"{stats['avg_quality_score']:.2f}")
    
    def data_analytics_ui(self):
        st.header("📊 Data Analytics")
        
        # Get analytics data
        analytics_data = self.db_ops.get_analytics_data()
        
        if analytics_data:
            df = pd.DataFrame(analytics_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Data types distribution
                type_counts = df['content_type'].value_counts()
                fig1 = px.pie(values=type_counts.values, names=type_counts.index,
                            title='Data Types Distribution')
                st.plotly_chart(fig1)
            
            with col2:
                # Quality scores over time
                df['created_at'] = pd.to_datetime(df['created_at'])
                fig2 = px.line(df, x='created_at', y='quality_score',
                             title='Quality Scores Over Time')
                st.plotly_chart(fig2)
            
            # Detailed analytics table
            st.subheader("Detailed Analytics")
            st.dataframe(df)
        
        else:
            st.info("No analytics data available. Run ETL pipeline first.")
    
    def ml_ui(self):
        st.header("🤖 Machine Learning")
        
        tab1, tab2, tab3 = st.tabs(["Classification", "Regression", "Clustering"])
        
        with tab1:
            st.subheader("Classification Models")
            # Placeholder for classification UI
            st.info("Upload labeled data to train classification models")
            
            # Example classification demo
            if st.button("Run Classification Demo"):
                try:
                    # Create sample data
                    from sklearn.datasets import make_classification
                    X, y = make_classification(n_samples=100, n_features=4, 
                                             n_classes=2, random_state=42)
                    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                    X_df = pd.DataFrame(X, columns=feature_names)
                    
                    # Train model
                    result = self.ml_manager.train_classification_model(X_df, pd.Series(y))
                    
                    # Display results
                    st.subheader("Classification Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Accuracy", f"{result['metrics']['accuracy']:.4f}")
                        st.metric("Precision", f"{result['metrics']['precision']:.4f}")
                    
                    with col2:
                        st.metric("Recall", f"{result['metrics']['recall']:.4f}")
                        st.metric("F1 Score", f"{result['metrics']['f1_score']:.4f}")
                    
                    # Feature importance
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        'feature': list(result['feature_importance'].keys()),
                        'importance': list(result['feature_importance'].values())
                    }).sort_values('importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='feature', y='importance', 
                               title='Feature Importance')
                    st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Demo failed: {str(e)}")
        
        with tab2:
            st.subheader("Regression Models")
            # Placeholder for regression UI
            st.info("Upload numerical data to train regression models")
            
            if st.button("Run Regression Demo"):
                try:
                    # Create sample data
                    from sklearn.datasets import make_regression
                    X, y = make_regression(n_samples=100, n_features=3, 
                                         noise=0.1, random_state=42)
                    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                    X_df = pd.DataFrame(X, columns=feature_names)
                    
                    # Train model
                    result = self.ml_manager.train_regression_model(X_df, pd.Series(y))
                    
                    # Display results
                    st.subheader("Regression Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("R² Score", f"{result['metrics']['r2_score']:.4f}")
                    
                    with col2:
                        st.metric("MSE", f"{result['metrics']['mse']:.4f}")
                    
                    with col3:
                        st.metric("RMSE", f"{result['metrics']['rmse']:.4f}")
                    
                except Exception as e:
                    st.error(f"Demo failed: {str(e)}")
        
        with tab3:
            st.subheader("Clustering Models")
            # Placeholder for clustering UI
            st.info("Upload data to discover patterns through clustering")
            
            if st.button("Run Clustering Demo"):
                try:
                    # Create sample data
                    from sklearn.datasets import make_blobs
                    X, _ = make_blobs(n_samples=100, centers=3, 
                                    n_features=2, random_state=42)
                    feature_names = ['feature_1', 'feature_2']
                    X_df = pd.DataFrame(X, columns=feature_names)
                    
                    # Train model
                    result = self.ml_manager.train_clustering_model(X_df, n_clusters=3)
                    
                    # Display results
                    st.subheader("Clustering Results")
                    
                    # Create scatter plot with clusters
                    X_df['cluster'] = result['clusters']
                    fig = px.scatter(X_df, x='feature_1', y='feature_2', 
                                   color='cluster', title='Cluster Visualization')
                    st.plotly_chart(fig)
                    
                    st.metric("Inertia", f"{result['metrics']['inertia']:.2f}")
                    st.metric("Number of Clusters", result['metrics']['n_clusters'])
                    
                except Exception as e:
                    st.error(f"Demo failed: {str(e)}")
    
    def rag_ui(self):
        st.header("🔍 RAG System (Retrieval Augmented Generation)")
        
        if not self.rag_system:
            st.warning("⚠️ Gemini API key not configured. RAG system unavailable.")
            st.info("Please set GEMINI_API_KEY in your .env file to enable RAG features.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query = st.text_area("Enter your question:", height=100, 
                               placeholder="Ask anything about your ingested data...")
            
            if st.button("Generate Answer"):
                if query:
                    with st.spinner("Searching and generating answer..."):
                        try:
                            # Create vector store if not exists
                            if self.rag_system.index is None:
                                with st.info("Building vector store from your data..."):
                                    self.rag_system.create_vector_store()
                            
                            result = self.rag_system.query_rag(query)
                            
                            st.subheader("Answer:")
                            st.write(result['response'])
                            
                            # Show sources
                            with st.expander("View Source Chunks"):
                                for i, chunk_info in enumerate(result['source_chunks']):
                                    st.write(f"**Chunk {i+1}:**")
                                    st.write(chunk_info['chunk']['chunk_text'][:500] + "...")
                                    st.write(f"Similarity Score: {1/(1 + chunk_info['distance']):.4f}")
                                    st.markdown("---")
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a question")
        
        with col2:
            st.subheader("RAG Configuration")
            chunk_size = st.slider("Chunk Size", 100, 1000, 512)
            top_k = st.slider("Top K Results", 1, 10, 5)
            
            if st.button("Rebuild Vector Store"):
                with st.spinner("Rebuilding vector store..."):
                    try:
                        self.rag_system.create_vector_store()
                        st.success("Vector store rebuilt successfully!")
                    except Exception as e:
                        st.error(f"Error rebuilding vector store: {str(e)}")
    
    def admin_dashboard_ui(self):
        if not st.session_state.get('admin_authenticated', False):
            st.warning("🔒 Please login as admin to access this section")
            return
        
        st.header("👨‍💼 Admin Dashboard")
        
        # System Overview
        st.subheader("System Overview")
        system_stats = self.db_ops.get_system_stats()
        
        if system_stats:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Data Sources", system_stats['total_sources'])
            col2.metric("Raw Data Records", system_stats['raw_data_count'])
            col3.metric("Processed Records", system_stats['processed_data_count'])
            col4.metric("Vector Chunks", system_stats['chunk_count'])
        
        # Data Sources Management
        st.subheader("Data Sources Management")
        sources = self.db_ops.get_all_sources()
        if sources:
            df_sources = pd.DataFrame(sources)
            st.dataframe(df_sources)
            
            # Export option
            if st.button("Export Sources to CSV"):
                csv = df_sources.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="data_sources.csv",
                    mime="text/csv"
                )
        
        # Database Operations
        st.subheader("Database Operations")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear All Data", type="secondary"):
                if st.checkbox("I understand this will delete all data"):
                    try:
                        self.db_ops.clear_all_data()
                        st.success("All data cleared successfully!")
                    except Exception as e:
                        st.error(f"Error clearing data: {str(e)}")
        
        with col2:
            if st.button("Run Data Validation"):
                with st.spinner("Running data validation..."):
                    try:
                        validation_results = self.db_ops.run_data_validation()
                        st.success("Data validation completed!")
                        
                        for table, result in validation_results.items():
                            with st.expander(f"{table} Validation"):
                                st.json(result)
                    
                    except Exception as e:
                        st.error(f"Validation failed: {str(e)}")

def main():
    app = DataPlatformApp()
    app.run()

if __name__ == "__main__":
    main()