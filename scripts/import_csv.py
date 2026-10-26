"""Import a measurement CSV into the Terrain Mapper database.

Lets you try the desktop app without the Raspberry Pi device. It mirrors what
the device does on upload: creates a session_<name> table with PostGIS points,
registers the session in measurement_sessions and computes its centroid.

Expected CSV columns (case-insensitive): time, latitude, longitude, altitude.
If `time` holds only HH:MM:SS, pass the measurement date with --date.

Usage:
    python scripts/import_csv.py data/sample_data.csv --session sample_data --date 2024-10-19
"""
import argparse
import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'desktop'))
from psycopg2 import sql  # noqa: E402
from db_connection import DatabaseConnection  # noqa: E402


def read_rows(path, date):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip().lower() for name in reader.fieldnames]
        for row in reader:
            t = row['time'].strip()
            ts = datetime.fromisoformat(f"{date} {t}" if date and len(t) <= 8 else t)
            yield ts, float(row['longitude']), float(row['latitude']), float(row['altitude'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv_path')
    parser.add_argument('--session', help='session name (default: CSV file name)')
    parser.add_argument('--date', help='measurement date YYYY-MM-DD, if the CSV holds only times')
    args = parser.parse_args()

    session = (args.session or os.path.splitext(os.path.basename(args.csv_path))[0]).lower().replace(' ', '_')
    table = sql.Identifier(f'session_{session}')
    rows = list(read_rows(args.csv_path, args.date))

    with DatabaseConnection() as db, db.conn, db.conn.cursor() as cur:
        cur.execute(sql.SQL("""
            CREATE TABLE {} (
                id SERIAL PRIMARY KEY,
                measurement_time TIMESTAMP WITHOUT TIME ZONE,
                location GEOGRAPHY(POINT, 4326),
                altitude NUMERIC(10, 2)
            )""").format(table))
        cur.executemany(
            sql.SQL("""INSERT INTO {} (measurement_time, location, altitude)
                       VALUES (%s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s)""").format(table),
            rows)
        cur.execute(sql.SQL("CREATE INDEX {} ON {} USING GIST (location)").format(
            sql.Identifier(f'idx_session_{session}_location'), table))
        cur.execute(sql.SQL("""
            INSERT INTO measurement_sessions (session_name, start_time, end_time, location_name, area_centroid)
            SELECT %s, MIN(measurement_time), MAX(measurement_time), %s,
                   ST_Centroid(ST_Collect(location::geometry))::geography
            FROM {}""").format(table), (session, session))

    print(f"Imported {len(rows)} points into session_{session}")


if __name__ == '__main__':
    main()
