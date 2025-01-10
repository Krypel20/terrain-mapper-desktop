# db_connection.py
import psycopg2
from psycopg2 import sql
import pandas as pd

class DatabaseConnection:
    def __init__(self, dbname, user='postgres', password=None, host='localhost', port=5433):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None

    def connect(self):
        if self.conn is None:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_sessions(self):
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT session_name FROM measurement_sessions ORDER BY start_time DESC")
        sessions = cur.fetchall()
        cur.close()
        return [session[0] for session in sessions]

    def get_session_data(self, session_name):
        table_name = f"session_{session_name.lower().replace(' ', '_')}"
        query = sql.SQL("""
            SELECT measurement_time, 
                ST_X(location::geometry) as longitude, 
                ST_Y(location::geometry) as latitude, 
                altitude 
            FROM {}
        """)
        
        with self.connect() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(query.format(sql.Identifier(table_name)))
                except Exception as e:
                    print(f"Error in get_session_data: {e}")
                    return pd.DataFrame()
                data = cur.fetchall()
                return pd.DataFrame(data, columns=['Time', 'Longitude', 'Latitude', 'Altitude'])
    
    def get_session_groups(self):
        """Groups sessions based on their geographic proximity using PostGIS."""
        conn = self.connect()
        cur = conn.cursor()

        query = """
        WITH groups AS (
            SELECT 
                m1.session_name,
                m1.area_centroid,
                MIN(m1.start_time) as group_time,
                ARRAY_AGG(m2.session_name ORDER BY m2.start_time DESC) as group_sessions
            FROM measurement_sessions m1
            JOIN measurement_sessions m2 ON ST_DWithin(
                m1.area_centroid::geography, 
                m2.area_centroid::geography,
                1000  -- 1km w metrach
            )
            WHERE m1.area_centroid IS NOT NULL 
            AND m2.area_centroid IS NOT NULL
            GROUP BY m1.session_name, m1.area_centroid
        )
        SELECT group_sessions, MIN(group_time) as min_time
        FROM groups
        GROUP BY group_sessions
        ORDER BY min_time DESC;
        """
        
        try:
            cur.execute(query)
            groups = [group[0] for group in cur.fetchall()]
            return groups
        except Exception as e:
            print(f"Error in get_session_groups: {e}")
            return []
        finally:
            cur.close()
    
    def delete_session(self, session_name):
        conn = self.connect()
        cur = conn.cursor()
        table_name = f"session_{session_name.lower().replace(' ', '_')}"
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name)))
        cur.execute(sql.SQL("DELETE FROM measurement_sessions WHERE session_name = %s"), (session_name,))
        conn.commit()
        cur.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()