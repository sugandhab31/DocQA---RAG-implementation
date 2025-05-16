from fastapi import FastAPI, UploadFile, File
import uvicorn
import aiofiles
import os
from app.utils.pdf_parser import extract_text_from_pdf
from app.utils.indexer import index_text_data

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok = True)


@app.get("/")
def read_root():
    return {
        "message": "Welcome to Document QA!"
    }

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    filepath = f"{UPLOAD_DIR}/{file.filename}"

    async with aiofiles.open(filepath, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    extracted_text = extract_text_from_pdf(filepath)
    os.remove(filepath)

    doc_id = file.filename.replace(".pdf", "")
    index = index_text_data(doc_id, extracted_text)

    return {
        "filename": file.filename,
        "text_preview": extracted_text[:500],
        "index_created": True
    }

if __name__ == "__main__":
    uvicorn.run("app.main.app", host="127.0.0.1", port=8000, reload=True)