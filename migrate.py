"""
migrate.py - ChromaDB → SQLite (sqlite-vec + FTS5) マイグレーション

既存ChromaDBから全チャンク(text, metadata, embeddings)を取得し、
SQLiteの統合DBに移行する。ChromaDBのデータは変更しない。
"""

import json
import sqlite3
import struct
import sys
from pathlib import Path

import chromadb
import sqlite_vec

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = DATA_DIR
SQLITE_PATH = DATA_DIR / "memory.sqlite3"
COLLECTION_NAME = "claude_sessions"
EMBEDDING_DIM = 384  # e5-small


def serialize_f32(vec: list[float]) -> bytes:
    """float list → little-endian bytes for sqlite-vec"""
    return struct.pack(f"<{len(vec)}f", *vec)


def create_schema(db: sqlite3.Connection):
    """SQLiteスキーマを作成"""
    db.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            session_id TEXT NOT NULL,
            project_path TEXT DEFAULT '',
            timestamp TEXT DEFAULT '',
            timestamp_epoch REAL DEFAULT 0.0,
            chunk_index INTEGER DEFAULT 0
        )
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_session
        ON chunks(session_id)
    """)
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec
        USING vec0(
            chunk_id TEXT PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}]
        )
    """)
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(
            chunk_id,
            text,
            tokenize='trigram'
        )
    """)
    db.commit()


def migrate():
    """ChromaDB → SQLite マイグレーション"""
    # ChromaDB読み取り
    print("ChromaDBに接続中...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )
    total = collection.count()
    print(f"ChromaDBチャンク数: {total}")

    if total == 0:
        print("データなし。終了。")
        return

    # SQLite準備
    print(f"SQLite DB作成: {SQLITE_PATH}")
    db = sqlite3.connect(str(SQLITE_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)

    create_schema(db)

    # 既存データチェック
    existing = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if existing > 0:
        print(f"SQLiteに既に{existing}件あるよ。中断。")
        print("やり直すなら memory.sqlite3 を削除してから再実行して。")
        db.close()
        return

    # バッチ取得（ChromaDB getは大量データだとメモリを使うので分割）
    BATCH = 5000
    migrated = 0

    for offset in range(0, total, BATCH):
        print(f"  取得中: {offset} - {min(offset + BATCH, total)} / {total}")
        result = collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=BATCH,
            offset=offset,
        )

        ids = result["ids"]
        documents = result["documents"]
        metadatas = result["metadatas"]
        embeddings = result["embeddings"]

        for i in range(len(ids)):
            chunk_id = ids[i]
            text = documents[i] if documents[i] else ""
            meta = metadatas[i] if metadatas[i] else {}
            emb = embeddings[i]

            if not text or emb is None or len(emb) == 0:
                continue

            db.execute(
                "INSERT OR IGNORE INTO chunks (chunk_id, text, session_id, project_path, timestamp, timestamp_epoch, chunk_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk_id,
                    text,
                    meta.get("session_id", ""),
                    meta.get("project_path", ""),
                    meta.get("timestamp", ""),
                    meta.get("timestamp_epoch", 0.0),
                    meta.get("chunk_index", 0),
                ),
            )
            db.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, serialize_f32(emb)),
            )
            db.execute(
                "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
                (chunk_id, text),
            )
            migrated += 1

        db.commit()
        print(f"  コミット済み: {migrated} 件")

    db.close()
    print(f"\n完了: {migrated} / {total} チャンクを移行")
    print(f"SQLite DB: {SQLITE_PATH}")
    print(f"サイズ: {SQLITE_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    migrate()
