"""RAG tool for querying local documents.

This module defines the `RAG` tool based on FAISS and LangChain to support
retrieval-augmented generation on local documents, including index persistence
management and prompt chain building.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, ClassVar

from crewai.tools import BaseTool
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

# LangChain Core (prompt/chain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

@dataclass
class Settings:
    """Configuration parameters for vectorization and retrieval."""
    # FAISS persistence
    persist_dir: str = "faiss"
    # Text splitting
    chunk_size: int = 1500
    chunk_overlap: int = 300
    # Retriever (MMR)
    search_type: str = "mmr"        # "mmr" or "similarity"
    k: int = 5                      # final results
    fetch_k: int = 20               # initial candidates (for MMR)
    mmr_lambda: float = 0.6         # 0 = max diversification, 1 = max relevance
    # Embedding
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # LM Studio (OpenAI-compatible)
    lmstudio_model_env: str = "LMSTUDIO_MODEL"  # model name in LM Studio, via env var

class RAG(BaseTool):
    """RAG tool for querying local documents."""

    name: str = "RAG"
    description: str = "A tool to search results on documents"

    # Use ClassVar to indicate it's a class variable, not a model field
    SETTINGS: ClassVar[Settings] = Settings()

    def get_embeddings(self, settings: Settings) -> AzureOpenAIEmbeddings:  # noqa: ARG002
        """Initialize the Azure OpenAI embedding model."""
        azure_endpoint = os.getenv("AZURE_API_BASE")
        azure_api_key = os.getenv("AZURE_API_KEY")
        azure_api_version = os.getenv("AZURE_API_VERSION")
        
        if not azure_endpoint:
            raise ValueError("AZURE_API_BASE environment variable is not set")
        if not azure_api_key:
            raise ValueError("AZURE_API_KEY environment variable is not set")
        if not azure_api_version:
            raise ValueError("AZURE_API_VERSION environment variable is not set")
        
        return AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            openai_api_version=azure_api_version
        )

    def get_llm_from_lmstudio(self, settings: Settings) -> AzureChatOpenAI:  # noqa: ARG002
        """Initialize a chat model on Azure OpenAI."""
        load_dotenv()
        
        azure_endpoint = os.getenv("AZURE_API_BASE")
        azure_api_key = os.getenv("AZURE_API_KEY")
        azure_api_version = os.getenv("AZURE_API_VERSION")
        
        if not azure_endpoint:
            raise ValueError("AZURE_API_BASE environment variable is not set")
        if not azure_api_key:
            raise ValueError("AZURE_API_KEY environment variable is not set")
        if not azure_api_version:
            raise ValueError("AZURE_API_VERSION environment variable is not set")
            
        return AzureChatOpenAI(
            azure_deployment="gpt-4o",
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version,
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def load_docs(self, folder_path: str) -> List[Document]:
        """Load all .rst files in a folder as a list of LangChain Documents."""
        docs: List[Document] = []
        folder = Path(folder_path).expanduser().resolve()

        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"The documents folder does not exist or is not a directory: {folder}")

        rst_files = sorted(folder.rglob("*.rst"))

        if not rst_files:
            return docs

        for file in rst_files:
            try:
                text = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = file.read_text(encoding="latin-1")

            rel_src = str(file.relative_to(folder))
            docs.append(Document(page_content=text, metadata={"source": rel_src, "abs_path": str(file)}))

        return docs
    
    def split_documents(self, docs: List[Document], settings: Settings) -> List[Document]:
        """Split documents to optimize retrieval."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
        )
        return splitter.split_documents(docs)
    
    def build_faiss_vectorstore(self, chunks: List[Document], embeddings: AzureOpenAIEmbeddings, persist_dir: str) -> FAISS:
        """Build a FAISS index and save it in the persistence directory."""
        vs = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
    
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        vs.save_local(persist_dir)
        return vs
    
    def load_or_build_vectorstore(self, settings: Settings, embeddings: AzureOpenAIEmbeddings, docs: List[Document]) -> FAISS:
        """Load an existing FAISS index or build it from scratch."""
        persist_path = Path(settings.persist_dir)
        index_file = persist_path / "index.faiss"
        meta_file = persist_path / "index.pkl"
    
        if index_file.exists() and meta_file.exists():
            return FAISS.load_local(
                settings.persist_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
        
        if not docs:
            raise ValueError(
                "No .rst documents found to build the index. "
                f"Check the documents folder: {settings.docs_dir}"
            )
    
        chunks = self.split_documents(docs, settings)
        return self.build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)
    
    def make_retriever(self, vector_store: FAISS, settings: Settings):
        """Configure the FAISS retriever."""
        if settings.search_type == "mmr":
            return vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": settings.k,
                    "fetch_k": settings.fetch_k,
                    "lambda_mult": settings.mmr_lambda,
                },
            )
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )
    
    def build_rag_chain(self, llm, retriever):
        """Create the RAG chain retrieval -> prompt -> LLM."""
        system_prompt = (
            "You are an expert assistant. Answer in English. "
            "Use only the CONTENT provided in the context. "
            "If the information is not present, state that it is not available. "
            "Be concise, accurate, and technically correct."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            (
                "human",
                "Question:\n{question}\n\n"
                "Context (selected extracts):\n{context}\n\n"
                "Instructions:\n"
                "1) Answer only with information contained in the context.\n"
                "2) If the answer is not in the context, write: "
                "'Not in documentation'",
            ),
        ])
    
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain
    
    def rag_answer(self, question: str, chain) -> str:
        """Run the RAG chain for a single question."""
        return chain.invoke(question)
    
    
    def _run(self, query: str, run_manager=None) -> str:
        """Tool entry point: run RAG on the provided query."""
        load_dotenv()

        docs_dir = os.getenv("DOCS_DIR", self.SETTINGS.docs_dir)
        persist_dir = os.getenv("FAISS_DIR", self.SETTINGS.persist_dir)

        self.SETTINGS.docs_dir = docs_dir
        self.SETTINGS.persist_dir = persist_dir

        embeddings = self.get_embeddings(self.SETTINGS)
        llm = self.get_llm_from_lmstudio(self.SETTINGS)

        docs = self.load_docs(self.SETTINGS.docs_dir)

        vector_store = self.load_or_build_vectorstore(
            self.SETTINGS,
            embeddings,
            docs,
        )

        retriever = self.make_retriever(vector_store, self.SETTINGS)
        chain = self.build_rag_chain(llm, retriever)
        answer = self.rag_answer(query, chain)

        return answer
