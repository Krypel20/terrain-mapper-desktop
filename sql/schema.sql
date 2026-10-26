-- Terrain Mapper – database schema (PostgreSQL 16 + PostGIS 3)
--
-- The measuring device (terrain-mapper-rspi) creates one table per measurement
-- session and registers it in measurement_sessions. The desktop app only reads
-- these tables (and can delete whole sessions).

CREATE EXTENSION IF NOT EXISTS postgis;

-- Session metadata: one row per measurement session
CREATE TABLE IF NOT EXISTS measurement_sessions (
    id            SERIAL PRIMARY KEY,
    session_name  TEXT NOT NULL,              -- e.g. 191024_145528 (DDMMYY_HHMMSS)
    start_time    TIMESTAMP WITHOUT TIME ZONE,
    end_time      TIMESTAMP WITHOUT TIME ZONE,
    location_name TEXT,
    area_centroid GEOGRAPHY(POINT, 4326)      -- centroid of all session points, used for grouping
);

-- Spatial index used by ST_DWithin when grouping sessions by location
CREATE INDEX IF NOT EXISTS idx_sessions_location
    ON measurement_sessions USING GIST (area_centroid);

-- Per-session measurement table (created dynamically, one per session):
--
-- CREATE TABLE session_<DDMMYY_HHMMSS> (
--     id               SERIAL PRIMARY KEY,
--     measurement_time TIMESTAMP WITHOUT TIME ZONE,
--     location         GEOGRAPHY(POINT, 4326),
--     altitude         NUMERIC(10, 2),
--     vdop             NUMERIC(10, 2),   -- newer sessions only
--     baro_alt         NUMERIC(10, 2)    -- newer sessions only
-- );
-- CREATE INDEX idx_session_<DDMMYY_HHMMSS>_location ON session_<...> USING GIST (location);
