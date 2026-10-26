-- Fill area_centroid for sessions uploaded by device versions that did not set it.
-- Safe to run multiple times; only sessions with a NULL centroid are updated.
DO $$
DECLARE
    r RECORD;
    tbl TEXT;
BEGIN
    FOR r IN SELECT session_name FROM measurement_sessions WHERE area_centroid IS NULL LOOP
        tbl := 'session_' || lower(r.session_name);
        IF to_regclass(format('public.%I', tbl)) IS NOT NULL THEN
            EXECUTE format(
                'UPDATE measurement_sessions
                    SET area_centroid = (SELECT ST_Centroid(ST_Collect(location::geometry))::geography FROM %I)
                  WHERE session_name = %L', tbl, r.session_name);
        END IF;
    END LOOP;
END $$;
