import google.generativeai as genai
from IPython.display import HTML, Markdown, display
import os
from sqlalchemy import create_engine, text

import sqlite3

GOOGLE_API_KEY = os.environ["GOOGLE_AI_STUDIO_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Create a connection to the SQLite database
engine = create_engine("sqlite:///sample.db", echo=True)

conn = engine.connect()

# Example query to test the connection
result = conn.execute(text("SELECT sqlite_version();"))
for row in result:
    print(f"SQLite Version: {row[0]}")

create_products_table = """
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name VARCHAR(255) NOT NULL,
    price DECIMAL(10, 2) NOT NULL
);
"""
result = conn.execute(text(create_products_table))
print(result)

create_staff_table = """
CREATE TABLE IF NOT EXISTS staff (
    staff_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL
);
"""
result = conn.execute(text(create_staff_table))
print(result)

create_orders_table = """
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name VARCHAR(255) NOT NULL,
    staff_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    FOREIGN KEY (staff_id) REFERENCES staff (staff_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);
"""
result = conn.execute(text(create_orders_table))
print(result)

fill_products_table = """
INSERT INTO products (product_name, price) VALUES
    ('Laptop', 799.99),
    ('Keyboard', 129.99),
    ('Mouse', 29.99);
"""
result = conn.execute(text(fill_products_table))
print(result)

fill_staff_table = """
INSERT INTO staff (first_name, last_name) VALUES
    ('Alice', 'Smith'),
    ('Bob', 'Johnson'),
    ('Charlie', 'Williams');
"""
result = conn.execute(text(fill_staff_table))
print(result)

fill_orders_table = """
INSERT INTO orders (customer_name, staff_id, product_id) VALUES
    ('David Lee', 1, 1),
    ('Emily Chen', 2, 2),
    ('Frank Brown', 1, 3);
"""
result = conn.execute(text(fill_orders_table))
print(result)

# Query to get all table names
query = "SELECT name FROM sqlite_master WHERE type='table';"
result = conn.execute(text(query))
tables = [row[0] for row in result]
print("Tables in the database:", tables)

conn.commit()


db_file = "sample.db"
db_conn = sqlite3.connect(db_file)


def list_tables() -> list[str]:
    """Retrieve the names of all tables in the database."""
    # Include print logging statements so you can see when functions are being called.
    print(" - DB CALL: list_tables")

    cursor = db_conn.cursor()

    # Fetch the table names.
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    tables = cursor.fetchall()
    return [t[0] for t in tables]


print(list_tables())


def describe_table(table_name: str) -> list[tuple[str, str]]:
    """Look up the table schema.

    Returns:
        List of columns, where each entry is a tuple of (column, type).
    """
    print(f" - DB CALL: describe_table - {table_name}")

    cursor = db_conn.cursor()

    cursor.execute(f"PRAGMA table_info({table_name});")

    schema = cursor.fetchall()
    # [column index, column name, column type, ...]
    return [(col[1], col[2]) for col in schema]


print(describe_table("products"))


def execute_query(sql: str) -> list[list[str]]:
    """Execute a SELECT statement, returning the results."""
    print(f" - DB CALL: execute_query:\n{sql}")

    cursor = db_conn.cursor()

    cursor.execute(sql)
    return cursor.fetchall()


print(execute_query("select * from products"))

# These are the Python functions defined above.
db_tools = [list_tables, describe_table, execute_query]

instruction = """You are a helpful chatbot that can interact with an
SQL database for a computer store. You will take the users questions
and turn them into SQL queries using the tools available. Once you
have the information you need, you will answer the user's question using
the data returned. Use list_tables to see what tables are present,
describe_table to understand the schema, and execute_query to issue
an SQL SELECT query."""

model = genai.GenerativeModel(
    "models/gemini-1.5-flash-latest",
    tools=db_tools,
    system_instruction=instruction
)

# Define a retry policy. The model might make multiple consecutive calls automatically
# for a complex query, this ensures the client retries if it hits quota limits.
from google.api_core import retry

retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}

# Start a chat with automatic function calling enabled.
chat = model.start_chat(enable_automatic_function_calling=True)

resp = chat.send_message(
    "What is the cheapest product?",
    request_options=retry_policy
)
print(resp.text)

resp = chat.send_message("and how much is it?", request_options=retry_policy)
print(resp.text)

model = genai.GenerativeModel(
    "models/gemini-1.5-pro-latest", tools=db_tools, system_instruction=instruction
)

chat = model.start_chat(enable_automatic_function_calling=True)
response = chat.send_message(
    "Which salesperson sold the cheapest product?", request_options=retry_policy
)
print(response.text)

import textwrap


def print_chat_turns(chat):
    """Prints out each turn in the chat history, including function calls and responses."""
    for event in chat.history:
        print(f"{event.role.capitalize()}:")

        for part in event.parts:
            if txt := part.text:
                print(f'  "{txt}"')
            elif fn := part.function_call:
                args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
                print(f"  Function call: {fn.name}({args})")
            elif resp := part.function_response:
                print("  Function response:")
                print(textwrap.indent(str(resp), "    "))

        print()


print_chat_turns(chat)
