import sqlite3
import numpy as np

DB_NAME = 'attendance_system.db'
ENCODING_LENGTH = 128 
def initialize_database():
    """Creates the necessary tables: 'users' and 'attendance_log'."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_log (
            log_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            time_in TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database '{DB_NAME}' initialized successfully.")

def store_encoding(user_id, name, encoding):
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    encoding_str = ','.join(map(str, encoding))

    try:
        cursor.execute("INSERT INTO users (id, name, encoding) VALUES (?, ?, ?)",
                       (user_id, name, encoding_str))
        conn.commit()
        print(f"User {name} (ID: {user_id}) stored successfully.")
    except sqlite3.IntegrityError:
        print(f"Error: User ID {user_id} already exists.")
    finally:
        conn.close()

if __name__ == "__main__":
    initialize_database()
    dummy_encoding = np.random.rand(ENCODING_LENGTH)
    store_encoding(1, "GAURAV SONI", dummy_encoding)

    dummy_encoding2 = np.random.rand(ENCODING_LENGTH)
    store_encoding2(2, "Jaishnu gehlot", dummy_encoding2)
