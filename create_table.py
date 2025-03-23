from sqlalchemy import create_engine, text

# Define the database URL for a local SQLite database
DATABASE_URL = "sqlite:///local_database.db"

# Create an engine
engine = create_engine(DATABASE_URL)

# SQL statement to create the 'users' table if it doesn't exist
create_table_query = """
CREATE TABLE IF NOT EXISTS va (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


# Execute the query using the connection
def create_table():
    with engine.connect() as connection:
        connection.execute(text(create_table_query))
        print("Table 'users' created or already exists.")


if __name__ == "__main__":
    create_table()
