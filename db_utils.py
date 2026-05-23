import os
import psycopg
from psycopg.rows import dict_row 
from datetime import datetime

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db_connection():
    if not DATABASE_URL:
        raise ValueError("Database environment variable missing")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def create_application_logs():
    with get_db_connection() as connection:
        # Psycopg 3 supports direct execution on connection!
        connection.execute('''CREATE TABLE IF NOT EXISTS app_log
                      (id SERIAL PRIMARY KEY,
                      session_id TEXT,
                      user_query  TEXT,
                      gpt_response  TEXT,
                      model TEXT,
                      created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                      )''')

def create_document_store():
    with get_db_connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id SERIAL PRIMARY KEY,
                     filename TEXT,
                     upload_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)''')

def insert_application_logs(session_id, user_query, gpt_response, model):
    with get_db_connection() as connection:
        connection.execute(
            'INSERT INTO app_log (session_id, user_query, gpt_response, model) VALUES (%s, %s, %s, %s)',
            (session_id, user_query, gpt_response, model)
        )

def get_chat_history(session_id):
    with get_db_connection() as connection:
        # Fixed: Removed the trailing comma from the end of the line below
        cursor = connection.execute(
            'SELECT user_query, gpt_response FROM app_log WHERE session_id=%s ORDER BY created_at',
            (session_id,)
        )
        messages = []
        for row in cursor.fetchall():
            messages.extend([
                {'role': 'human', 'content': row['user_query']},
                {'role': 'ai', 'content': row['gpt_response']}
            ])
        return messages

def insert_document_record(filename):
    with get_db_connection() as conn:
        # To get the RETURNING id data, we capture the returned cursor from .execute()
        cursor = conn.execute('INSERT INTO document_store (filename) VALUES (%s) RETURNING id', (filename,))
        file_id = cursor.fetchone()['id']
        return file_id

def delete_document_record(file_id):
    with get_db_connection() as conn:
        conn.execute('DELETE FROM document_store WHERE id = %s', (file_id,))
        return True

def get_all_documents():
    with get_db_connection() as conn:
        # Capture the cursor to fetch the list of documents
        cursor = conn.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
        documents = cursor.fetchall()
        return documents

# Initialize the database tables
if __name__ == "__main__":
    create_application_logs()
    create_document_store()