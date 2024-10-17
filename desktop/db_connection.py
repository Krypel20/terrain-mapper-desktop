# db_connection.py
import psycopg2
from psycopg2 import sql
import pandas as pd

class DatabaseConnection:
    def __init__(self, dbname, user='postgres', password=None, host='localhost', port=5432):
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
        conn = self.connect()
        cur = conn.cursor()
        table_name = f"measurements_{session_name.lower().replace(' ', '_')}"
        query = sql.SQL("""
            SELECT measurement_time, 
                   ST_X(location::geometry) as longitude, 
                   ST_Y(location::geometry) as latitude, 
                   altitude 
            FROM {}
        """)
        cur.execute(query.format(sql.Identifier(table_name)))
        data = cur.fetchall()
        cur.close()
        return pd.DataFrame(data, columns=['Time', 'Longitude', 'Latitude', 'Altitude'])

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()