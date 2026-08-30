-- Research notes table for saving user analysis and research
CREATE TABLE IF NOT EXISTS research_notes (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    note_type VARCHAR(20) NOT NULL CHECK (note_type IN ('analysis', 'thesis', 'alert', 'general')),
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_research_notes_email ON research_notes(email);
CREATE INDEX IF NOT EXISTS idx_research_notes_ticker ON research_notes(ticker);
CREATE INDEX IF NOT EXISTS idx_research_notes_created ON research_notes(created_at DESC);

-- User visits table for tracking last visit times (for alerts feature)
CREATE TABLE IF NOT EXISTS user_visits (
    email VARCHAR(255) PRIMARY KEY,
    last_visit_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    previous_visit_at TIMESTAMP
);

-- Update user visit timestamp helper
-- This would typically be called on each app login/visit
CREATE OR REPLACE FUNCTION update_user_visit(user_email VARCHAR(255))
RETURNS VOID AS $$
BEGIN
    INSERT INTO user_visits (email, last_visit_at, previous_visit_at)
    VALUES (user_email, CURRENT_TIMESTAMP, NULL)
    ON CONFLICT (email) DO UPDATE
    SET previous_visit_at = user_visits.last_visit_at,
        last_visit_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;
