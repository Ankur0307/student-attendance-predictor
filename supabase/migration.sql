-- ─────────────────────────────────────────────────────────────────
-- Supabase Migration — Student Attendance Predictive System
-- Run this ONCE in: Supabase Dashboard → SQL Editor → New Query
-- ─────────────────────────────────────────────────────────────────

-- Drop & recreate the attendance table with proper columns
DROP TABLE IF EXISTS attendance CASCADE;

CREATE TABLE attendance (
    id                      BIGSERIAL PRIMARY KEY,
    student_id              TEXT NOT NULL,
    student_name            TEXT,
    date                    DATE,
    subject_code            TEXT,
    subject_name            TEXT,
    faculty_id              TEXT,
    class_start_time        TEXT,
    class_end_time          TEXT,
    time_in                 TEXT,
    time_out                TEXT,
    semester                TEXT,
    status                  INTEGER DEFAULT 0,   -- 1 = Present, 0 = Absent
    is_exam_week            INTEGER DEFAULT 0,
    late_entry              INTEGER DEFAULT 0,
    remarks                 TEXT,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

-- Drop & recreate the students lookup table
DROP TABLE IF EXISTS students CASCADE;

CREATE TABLE students (
    student_id  TEXT PRIMARY KEY,
    name        TEXT,
    semester    TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Drop & recreate the subjects lookup table
DROP TABLE IF EXISTS subjects CASCADE;

CREATE TABLE subjects (
    subject_code  TEXT PRIMARY KEY,
    subject_name  TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ── Row Level Security ────────────────────────────────────────────
ALTER TABLE attendance ENABLE ROW LEVEL SECURITY;
ALTER TABLE students   ENABLE ROW LEVEL SECURITY;
ALTER TABLE subjects   ENABLE ROW LEVEL SECURITY;

-- Allow public READ access (anon key can SELECT)
CREATE POLICY "Allow public read on attendance" ON attendance
    FOR SELECT USING (true);

CREATE POLICY "Allow public read on students" ON students
    FOR SELECT USING (true);

CREATE POLICY "Allow public read on subjects" ON subjects
    FOR SELECT USING (true);

-- ── Indexes for fast lookups ──────────────────────────────────────
CREATE INDEX idx_attendance_student_id    ON attendance(student_id);
CREATE INDEX idx_attendance_subject_code  ON attendance(subject_code);
CREATE INDEX idx_attendance_date          ON attendance(date);

-- Done!
SELECT 'Migration complete ✅' AS result;
