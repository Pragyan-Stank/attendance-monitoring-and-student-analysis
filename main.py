from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import database as db
from pipeline import AsyncPipeline
import traceback
from starlette.concurrency import run_in_threadpool
import base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipeline = AsyncPipeline()

class RegisterRequest(BaseModel):
    student_id: str
    name: str
    image_b64: str

class SessionStartRequest(BaseModel):
    label: str

class FrameRequest(BaseModel):
    session_id: str
    frame_b64: str

@app.on_event("startup")
async def startup_event():
    db.init_db()
    # Pre-warm AI pipelines
    await run_in_threadpool(pipeline.initialize)

@app.get("/api/status")
def get_ai_status():
    return {"initialized": pipeline._initialized}

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.get("/api/students")
def get_students():
    return db.get_all_students()

@app.delete("/api/students/{student_id}")
def delete_student(student_id: str):
    db.delete_student(student_id)
    return {"status": "deleted"}

@app.post("/api/register")
async def register_student(req: RegisterRequest):
    try:
        data = req.image_b64.split(",")[1] if "," in req.image_b64 else req.image_b64
        embedding, _ = await run_in_threadpool(pipeline.capture_enrollment, data)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face found. Make sure you are clearly visible.")
        
        db.add_student(req.student_id, req.name, embedding, req.image_b64)
        # Reload DB in pipeline
        await run_in_threadpool(pipeline.load_student_db)
        return {"status": "enrolled"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/start")
def start_session(req: SessionStartRequest):
    session_id = str(uuid.uuid4())
    db.start_session(session_id, req.label)
    pipeline.start_session(session_id)
    return {"session_id": session_id}

@app.post("/api/session/end")
def end_session(req: dict):
    session_id = req.get("session_id")
    db.end_session(session_id)
    report = db.get_session_report(session_id)
    return {"report": {"students": report}}

@app.post("/api/process_frame")
async def process_frame(req: FrameRequest):
    try:
        data = req.frame_b64.split(",")[1] if "," in req.frame_b64 else req.frame_b64
        results = await run_in_threadpool(pipeline.process_frame, data)
        return {"students": results}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
