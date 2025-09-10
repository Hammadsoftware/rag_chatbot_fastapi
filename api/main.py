from fastapi import FastAPI, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from logic import rag_pipeline, query_pipe_line
import os
import uuid

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload/")
async def upload_file(file: UploadFile):
    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, temp_filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = rag_pipeline(file_path)
    return {"message": "File uploaded successfully", "rag_result": result}


# ✅ GET with query param
@app.get("/query/")
async def query_piple(query: str = Query(..., description="User's question")):
    result = query_pipe_line(query)
    return result
