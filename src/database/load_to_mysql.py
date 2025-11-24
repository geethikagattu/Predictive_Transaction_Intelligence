import pandas as pd
from mysql_connection import get_mysql_connection

def load_csv_to_mysql(csv_path):
    df = pd.read_csv(csv_path)

    conn = get_mysql_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            transaction_id VARCHAR(50),
            amount DOUBLE,
            timestamp DATETIME,
            merchant VARCHAR(100),
            location VARCHAR(100),
            label INT
        );
    """)

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO transactions
            (transaction_id, amount, timestamp, merchant, location, label)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, tuple(row))

    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    load_csv_to_mysql("data/processed/train.csv")
