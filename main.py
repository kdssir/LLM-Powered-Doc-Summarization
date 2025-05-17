from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from summarizer import DocumentSummarizer
import shutil
import os
from fastapi.templating import Jinja2Templates

app = FastAPI()
chatbot = None

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Serve HTML templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDF files are supported."})

    file_path = os.path.join("data", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    global chatbot
    chatbot = RAGChatbot(file_path)
    print(chatbot)
    return {"message": f"{file.filename} uploaded and indexed successfully."}


@app.post("/summarize/")
async def summarize_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    pdf_path = os.path.join("data", file.filename)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    summarizer = DocumentSummarizer(pdf_path)
    summary = summarizer.summarize()
    return JSONResponse(content={"summary": summary})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
