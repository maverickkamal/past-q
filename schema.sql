-- Medical Past Questions (PQ) Pipeline SQLite Schema
-- System Version: 2.1

-- Parent exam papers
CREATE TABLE IF NOT EXISTS exams (
    id TEXT PRIMARY KEY,                       -- e.g., 'ANA_2021_PROF_P1'
    source_document TEXT NOT NULL,             -- Source PDF file name
    academic_year TEXT,                        -- e.g., '2021/2022'
    discipline TEXT NOT NULL,                  -- Anatomy, Physiology, Biochemistry
    level TEXT DEFAULT 'UNKNOWN',              -- 200L, 300L, UNKNOWN
    paper_title TEXT NOT NULL,                 -- e.g., 'Paper I (Gross Anatomy)'
    exam_type TEXT,                            -- Professional MBBS, In-Course Assessment
    examiner TEXT DEFAULT 'UNKNOWN',           -- Primary lecturer for the paper
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Unified question table
CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,                       -- e.g., 'ANA_2021_PROF_P1_Q1'
    exam_id TEXT NOT NULL REFERENCES exams(id),
    question_number TEXT NOT NULL,             -- e.g., '1', '2(b)', 'Station 4'
    category TEXT NOT NULL,                    -- ESSAY | OBJECTIVE | STEEPLECHASE | CA_QUIZ
    sub_type TEXT NOT NULL,                    -- SEQ | SAQ | SBA | MULTIPLE_TRUE_FALSE | SPOTTER | IN_COURSE_TEST | QUIZ
    discipline TEXT NOT NULL,                  -- Anatomy | Physiology | Biochemistry
    level TEXT DEFAULT 'UNKNOWN',              -- 200L | 300L | UNKNOWN
    course_code TEXT,                          -- e.g. ANA 201a, PIO 205
    system_region TEXT NOT NULL,               -- Canonical organ system e.g. Thorax and Abdomen, Lower Limb
    topic TEXT NOT NULL,                       -- e.g., Brachial Plexus, Beta-Oxidation
    examiner TEXT DEFAULT 'UNKNOWN',           -- Section lecturer
    curriculum_style TEXT NOT NULL,            -- BMAS_RECALL | CCMAS_VIGNETTE
    stem_text TEXT NOT NULL,                   -- Clinical case or question prompt
    items_json TEXT,                           -- JSON serialized QuestionItem array
    total_marks INTEGER,                       -- Stated total marks (nullable)
    has_diagram BOOLEAN DEFAULT 0,
    diagram_path TEXT,                         -- Local path to WebP asset
    review_status TEXT DEFAULT 'APPROVED',     -- APPROVED | NEEDS_REVIEW
    flag_reasons TEXT,                         -- Diagnostic validation messages
    recurrence_count INTEGER DEFAULT 1,        -- Number of times similar question appeared across sessions
    recurrence_cluster_id TEXT,                -- Deterministic cluster ID for recurring questions
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for targeted study queries
CREATE INDEX IF NOT EXISTS idx_questions_study_filter
ON questions (level, discipline, category, system_region);

CREATE INDEX IF NOT EXISTS idx_questions_taxonomy 
ON questions (discipline, system_region, topic);

CREATE INDEX IF NOT EXISTS idx_questions_examiner 
ON questions (examiner);

CREATE INDEX IF NOT EXISTS idx_questions_status 
ON questions (review_status);

CREATE INDEX IF NOT EXISTS idx_questions_recurrence
ON questions (recurrence_cluster_id);
