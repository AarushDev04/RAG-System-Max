-- ============================================================
-- Metadata Store Schema
-- Replaces: Amazon RDS for PostgreSQL
-- Run automatically by the postgres Docker container on first boot
-- ============================================================

-- ----------------------------------------------------------------
-- documents: one row per ingested document
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id      VARCHAR(64)  PRIMARY KEY,           -- uuid5 of the file path
    filename    TEXT         NOT NULL,
    num_chunks  INTEGER      NOT NULL DEFAULT 0,
    status      VARCHAR(16)  NOT NULL DEFAULT 'pending',
                                                    -- pending | ingested | failed
    ingested_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    metadata    JSONB,                              -- arbitrary extra fields
    CONSTRAINT  status_check CHECK (status IN ('pending', 'ingested', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_documents_ingested ON documents(ingested_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_status   ON documents(status);

-- ----------------------------------------------------------------
-- query_logs: one row per /query call (optional — for analytics)
-- Written by the orchestrator after every successful generation.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_logs (
    id          BIGSERIAL    PRIMARY KEY,
    session_id  VARCHAR(64)  NOT NULL DEFAULT 'default',
    query       TEXT         NOT NULL,
    query_type  VARCHAR(32),                        -- simple_factoid | multi_hop | etc.
    cache_type  VARCHAR(16),                        -- exact | semantic | miss
    answer      TEXT,
    latency_ms  DOUBLE PRECISION,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_session    ON query_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_created    ON query_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_query_logs_query_type ON query_logs(query_type);
