import sqlite3
import numpy as np

DB_PATH = "attendance.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop the JSON students table to revert to InsightFace BLOBs
    cursor.execute("DROP TABLE IF EXISTS students")
    
    # Students table
    cursor.execute('''CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY,
        name TEXT,
        embedding BLOB,
        photo_b64 TEXT,
        enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        label TEXT,
        start_time TIMESTAMP,
        end_time TIMESTAMP
    )''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        student_id TEXT,
        timestamp TIMESTAMP,
        confidence REAL,
        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS engagement_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        student_id TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_focused INTEGER,
        yaw REAL,
        pitch REAL,
        distraction_type TEXT
    )''')
    
    conn.commit()
    conn.close()

def add_student(student_id, name, embedding, photo_b64):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    emb_bytes = embedding.tobytes()
    cursor.execute(
        "INSERT OR REPLACE INTO students (student_id, name, embedding, photo_b64) VALUES (?, ?, ?, ?)",
        (student_id, name, emb_bytes, photo_b64)
    )
    conn.commit()
    conn.close()

def get_all_embeddings():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, name, embedding FROM students")
    rows = cursor.fetchall()
    results = []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        results.append({
            "student_id": r["student_id"],
            "name": r["name"],
            "embedding": emb
        })
    conn.close()
    return results

def get_all_students():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, name, photo_b64, enrolled_at FROM students")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def delete_student(student_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()

def start_session(session_id, label):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    import datetime
    cursor.execute(
        "INSERT INTO sessions (session_id, label, start_time) VALUES (?, ?, ?)",
        (session_id, label, datetime.datetime.now())
    )
    conn.commit()
    conn.close()

def mark_attendance(session_id, student_id, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    import datetime
    cursor.execute("SELECT id FROM attendance WHERE session_id=? AND student_id=?", (session_id, student_id))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO attendance (session_id, student_id, timestamp, confidence) VALUES (?, ?, ?, ?)",
            (session_id, student_id, datetime.datetime.now(), float(confidence))
        )
    conn.commit()
    conn.close()

def log_engagement(session_id, student_id, is_focused, yaw, pitch, distraction_type):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO engagement_logs (session_id, student_id, is_focused, yaw, pitch, distraction_type) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, student_id, int(is_focused), float(yaw), float(pitch), distraction_type)
    )
    conn.commit()
    conn.close()

def end_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    import datetime
    cursor.execute(
        "UPDATE sessions SET end_time = ? WHERE session_id = ?",
        (datetime.datetime.now(), session_id)
    )
    conn.commit()
    conn.close()

def get_session_report(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT student_id, name FROM students")
    students = {r["student_id"]: r["name"] for r in cursor.fetchall()}
    
    cursor.execute("SELECT student_id FROM attendance WHERE session_id = ?", (session_id,))
    present_ids = {r["student_id"] for r in cursor.fetchall()}
    
    report = []
    for sid, name in students.items():
        cursor.execute("""
            SELECT AVG(is_focused) as avg_eng, 
                   COUNT(CASE WHEN distraction_type IS NOT NULL THEN 1 END) as distract_count
            FROM engagement_logs 
            WHERE session_id = ? AND student_id = ?
        """, (session_id, sid))
        stats = cursor.fetchone()
        report.append({
            "student_id": sid,
            "name": name,
            "present": sid in present_ids,
            "avg_engagement": stats["avg_eng"] or 0,
            "distraction_count": stats["distract_count"] or 0
        })
    conn.close()
    return report
