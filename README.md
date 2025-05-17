# LLM Powered Document Summarization (LangChain + Transformers)

This project is a **document summarization tool** built using **LangChain**, **Hugging Face Transformers**, and **FastAPI**. It allows users to upload PDF documents and generate concise summaries using state-of-the-art language models.

---

## Features

*  PDF document upload support
*  Chunking of large documents using `RecursiveCharacterTextSplitter`
*  Summarization via HuggingFace's `facebook/bart-large-cnn` model
*  Two summarization modes:

  * `map_reduce` (default): summarization with global understanding
  * `refine`: incremental, paragraph-by-paragraph summarization
*  Page-wise summary option with metadata
*  Local summary caching using MD5 file hashing
*  Frontend UI built with HTML/CSS

---


##  Setup

### 1. Install dependencies

```bash
pip install fastapi uvicorn langchain transformers torch
pip install pypdf  # for PDF loading
```

### 2. Run the FastAPI backend

```bash
uvicorn main:app --reload
```

### 3. Open the Frontend UI

Go to: `http://localhost:8000`

---

##  Example Usage

1. Upload a PDF using the web form.
2. The backend will:

   * Load and split the PDF
   * Generate a summary using the selected strategy (`map_reduce` by default)
3. The summary appears on the same page.

---

##  API Endpoints

### POST `/summarize/`

Uploads a PDF and returns a document summary.

**Form Data:**

* `file`: PDF file

**Response:**

```json
{
  "summary": "..."
}
```

---

##  Future Improvements

* Add support for DOCX and TXT files
* Deploy with persistent vector storage (e.g., FAISS + SQLite)
* Add OpenAI/GPT model backend (via API key)
* Multi-language summarization
* User authentication and history

---

##  Author

Built as a practice project to demonstrate LangChain + Hugging Face summarization with an interactive UI.

---

## Acknowledgments

This project is made possible by the following open-source tools and libraries:

* [LangChain](https://github.com/langchain-ai/langchain)
* [Transformers by Hugging Face](https://github.com/huggingface/transformers)
* [FastAPI](https://github.com/tiangolo/fastapi)
* [PyPDF](https://github.com/py-pdf/pypdf)
* [Facebook BART Large CNN Model](https://huggingface.co/facebook/bart-large-cnn)

Special thanks to the open-source community for providing these incredible tools!
