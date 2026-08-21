import sqlite3
import threading

DB_PATH = "volume_history.db"

_write_lock = threading.Lock()


def init_db(db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kamera TEXT NOT NULL,
                mulai TEXT NOT NULL,
                selesai TEXT NOT NULL,
                bus INTEGER NOT NULL,
                mobil INTEGER NOT NULL,
                motor INTEGER NOT NULL,
                truk INTEGER NOT NULL,
                total INTEGER NOT NULL
            )
            """
        )


def insert_history(entry, db_path=DB_PATH):
    """entry: dict with keys Camera, Start, End, Bus, Car, Motorcycle, Truck, Total
    (the CameraWorker.history format).

    The SQL column names stay in Indonesian on purpose: they are the existing
    storage format and server.py maps them to the public API keys, so renaming
    them would break both the stored data and the API contract.
    """
    with _write_lock, sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO history (kamera, mulai, selesai, bus, mobil, motor, truk, total)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["Camera"],
                entry["Start"],
                entry["End"],
                entry["Bus"],
                entry["Car"],
                entry["Motorcycle"],
                entry["Truck"],
                entry["Total"],
            ),
        )


def get_latest_per_kamera(db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT h.* FROM history h
            INNER JOIN (
                SELECT kamera, MAX(id) AS max_id FROM history GROUP BY kamera
            ) latest ON h.kamera = latest.kamera AND h.id = latest.max_id
            """
        ).fetchall()
        return [dict(r) for r in rows]


def get_history(kamera=None, limit=100, db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if kamera:
            rows = conn.execute(
                "SELECT * FROM history WHERE kamera = ? ORDER BY id DESC LIMIT ?",
                (kamera, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
