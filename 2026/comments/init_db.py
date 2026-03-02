import sqlite3
from datetime import datetime

conn = sqlite3.connect('comments.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        comment_text TEXT NOT NULL,
        normal REAL, insult REAL, threat REAL, obscenity REAL,
        main_class TEXT,
        is_toxic INTEGER,
        timestamp TEXT
    )
''')
conn.commit()
conn.close()
print("БД создана")