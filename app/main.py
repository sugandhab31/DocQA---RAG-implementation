from fastapi import FastAPI, UploadFile, File
import uvicorn
import aiofiles
import os
from app.utils.pdf_parser import extract_text_from_pdf

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

    return {
        "filename": file.filename,
        "text_preview": extracted_text[:500]  # Show first 500 chars
    }

if __name__ == "__main__":
    uvicorn.run("app.main.app", host="127.0.0.1", port=8000, reload=True)