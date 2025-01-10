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

        # Pobierz wszystkie sesje
        cur.execute("SELECT session_name, area_centroid FROM measurement_sessions WHERE area_centroid IS NOT NULL ORDER BY start_time DESC")
        all_sessions = cur.fetchall()
        
        groups = []
        used_sessions = set()
        
        for session_name, centroid in all_sessions:
            # Pomiń jeśli sesja jest już w jakiejś grupie
            if session_name in used_sessions:
                continue
                
            # Znajdź pobliskie sesje
            query = """
            SELECT session_name
            FROM measurement_sessions
            WHERE ST_DWithin(
                area_centroid::geography, 
                %s::geography,
                1000  -- 1km w metrach
            )
            AND area_centroid IS NOT NULL
            ORDER BY start_time DESC;
            """
            
            cur.execute(query, [centroid])
            nearby_sessions = [row[0] for row in cur.fetchall()]
            
            # Utwórz nową grupę tylko z sesjami, które nie są jeszcze użyte
            new_group = [s for s in nearby_sessions if s not in used_sessions]
            
            if new_group:  # Jeśli grupa nie jest pusta
                groups.append(new_group)
                used_sessions.update(new_group)
        
        cur.close()
        return groups

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