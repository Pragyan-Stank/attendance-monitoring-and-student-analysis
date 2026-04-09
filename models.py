from pydantic import BaseModel
from typing import Optional, List

class RegisterRequest(BaseModel):
    student_id: str
    name: str
    image_b64: str

class SessionStartRequest(BaseModel):
    label: str

class SessionEndRequest(BaseModel):
    session_id: str

class ProcessFrameRequest(BaseModel):
    session_id: str
    frame_b64: str

class StudentReport(BaseModel):
    student_id: str
    name: str
    present: bool
    marked_at: Optional[str] = None
    confidence: Optional[float] = None
    avg_engagement: Optional[float] = None
    distraction_count: int

class SessionReport(BaseModel):
    session_id: str
    label: Optional[str] = None
    started_at: str
    ended_at: Optional[str] = None
    students: List[StudentReport]
