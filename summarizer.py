from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import hashlib
import os
import json


class DocumentSummarizer:
    def __init__(self, pdf_path: str, cache_dir: str = "summary_cache"):
        self.pdf_path = pdf_path
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.doc_hash = self._get_file_hash(pdf_path)
        self.cache_path = os.path.join(self.cache_dir, f"{self.doc_hash}.json")

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = splitter.split_documents(docs)

        summarizer_pipe = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn"
        )
        self.llm = HuggingFacePipeline(pipeline=summarizer_pipe)

    def _get_file_hash(self, path: str) -> str:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def summarize(self, mode: str = "detailed") -> str:
        # Check for cache
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                return json.load(f).get(mode, "")

        try:
            chain_type = "map_reduce" if mode == "detailed" else "refine"
            chain = load_summarize_chain(self.llm, chain_type=chain_type)
            summary = chain.run(self.chunks)

            # Save to cache
            with open(self.cache_path, 'w') as f:
                json.dump({mode: summary}, f)

            return summary
        except Exception as e:
            print(f"Summarization error: {e}")
            return "Sorry, summarization failed."

    def summarize_by_page(self) -> dict:
        """
        Summarizes each page individually and returns a dict: {page_number: summary}
        """
        try:
            page_summaries = {}
            for i, chunk in enumerate(self.chunks):
                page = chunk.metadata.get("page", i + 1)
                result = self.llm(chunk.page_content)
                page_summaries[f"Page {page}"] = result[0]['summary_text']
            return page_summaries
        except Exception as e:
            print(f"Page-wise summarization error: {e}")
            return {"error": "Failed to summarize by page."}
