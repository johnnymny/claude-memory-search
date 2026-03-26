"""
ingest.py - Claude Code セッションログを SQLite (sqlite-vec + FTS5) に格納する

jsonlファイルをパースし、user/assistantペアのスライディングウィンドウで
チャンク分割、multilingual-e5-small でembeddingして SQLite に保存する。
増分更新対応: 既にインデックス済みのファイルはスキップする。
"""

import json
import os
import re
import hashlib
import sqlite3
import struct
from pathlib import Path
from datetime import datetime

import sqlite_vec
from sentence_transformers import SentenceTransformer

# --- 設定 ---
PROJECTS_DIR = Path.home() / ".claude" / "projects"
DATA_DIR = Path(__file__).parent / "data"
SQLITE_PATH = DATA_DIR / "memory.sqlite3"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
WINDOW_SIZE = 4  # user/assistantペア数
WINDOW_STEP = 2  # スライドステップ
MAX_CHUNK_CHARS = 2000  # チャンクの最大文字数

# 除外するメッセージtype
SKIP_TYPES = {"system", "progress", "file-history-snapshot", "queue-operation"}


def serialize_f32(vec: list[float]) -> bytes:
    """float list → little-endian bytes for sqlite-vec"""
    return struct.pack(f"<{len(vec)}f", *vec)


def get_db() -> sqlite3.Connection:
    """SQLite接続を返す（sqlite-vec拡張ロード済み）"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(SQLITE_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    # スキーマ作成（冪等）
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
    return db


def get_ingested_sessions(db: sqlite3.Connection) -> set[str]:
    """既にインデックス済みのsession_idを取得する"""
    rows = db.execute("SELECT DISTINCT session_id FROM chunks").fetchall()
    return {r[0] for r in rows}


def extract_text_from_content(content) -> str:
    """メッセージのcontentからテキストを抽出する"""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")

            if item_type == "text":
                text = item.get("text", "")
                # system-reminderタグを除去
                text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL)
                text = text.strip()
                if text:
                    parts.append(text)

            elif item_type == "tool_use":
                tool_name = item.get("name", "unknown")
                tool_input = item.get("input", {})
                summary_parts = []
                if isinstance(tool_input, dict):
                    for key in ["command", "query", "pattern", "file_path", "prompt", "url"]:
                        if key in tool_input:
                            val = str(tool_input[key])[:100]
                            summary_parts.append(f"{key}={val}")
                input_summary = ", ".join(summary_parts) if summary_parts else ""
                parts.append(f"[tool: {tool_name}({input_summary})]")

            elif item_type == "tool_result":
                result_content = item.get("content", "")
                if isinstance(result_content, str):
                    result_text = result_content[:150]
                elif isinstance(result_content, list):
                    texts = []
                    for rc in result_content:
                        if isinstance(rc, dict) and rc.get("type") == "text":
                            texts.append(str(rc.get("text", ""))[:100])
                    result_text = " ".join(texts)[:150]
                else:
                    result_text = ""
                if result_text:
                    parts.append(f"[result: {result_text}]")

        return " ".join(parts)

    return ""


def parse_jsonl(filepath: Path) -> list[dict]:
    """jsonlファイルからuser/assistantメッセージを抽出する"""
    messages = []
    session_id = filepath.stem

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type", "")

            if msg_type in SKIP_TYPES:
                continue

            if msg_type not in ("user", "assistant"):
                continue

            role = msg_type
            content = None

            if "message" in obj:
                msg = obj["message"]
                role = msg.get("role", msg_type)
                content = msg.get("content")
            elif "content" in obj:
                content = obj["content"]
            else:
                continue

            text = extract_text_from_content(content)
            if not text or len(text.strip()) < 5:
                continue

            timestamp = obj.get("timestamp", "")

            messages.append({
                "role": role,
                "text": text,
                "timestamp": timestamp,
                "session_id": session_id,
            })

    return messages


def create_chunks(messages: list[dict], session_id: str, project_path: str) -> list[dict]:
    """user/assistantペアのスライディングウィンドウでチャンクを作る"""
    pairs = []
    i = 0
    while i < len(messages):
        if messages[i]["role"] == "user":
            user_msg = messages[i]
            assistant_msg = None
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                assistant_msg = messages[i + 1]
                i += 2
            else:
                i += 1
            pairs.append((user_msg, assistant_msg))
        else:
            i += 1

    if not pairs:
        return []

    chunks = []
    chunk_index = 0

    for start in range(0, len(pairs), WINDOW_STEP):
        end = min(start + WINDOW_SIZE, len(pairs))
        window = pairs[start:end]

        if not window:
            break

        lines = []
        for user_msg, assistant_msg in window:
            user_text = user_msg["text"][:500]
            lines.append(f"User: {user_text}")
            if assistant_msg:
                assistant_text = assistant_msg["text"][:500]
                lines.append(f"Assistant: {assistant_text}")

        chunk_text = "\n".join(lines)
        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunk_text = chunk_text[:MAX_CHUNK_CHARS]

        timestamp = window[0][0].get("timestamp", "")

        chunk_id = hashlib.md5(
            f"{session_id}:{chunk_index}".encode()
        ).hexdigest()

        timestamp_epoch = 0.0
        if timestamp:
            try:
                dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                timestamp_epoch = dt.timestamp()
            except Exception:
                pass

        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "session_id": session_id,
                "project_path": project_path,
                "timestamp": str(timestamp),
                "timestamp_epoch": timestamp_epoch,
                "chunk_index": chunk_index,
            },
        })
        chunk_index += 1

    return chunks


def find_jsonl_files() -> list[tuple[Path, str]]:
    """全プロジェクトのjsonlファイルを探す"""
    results = []
    if not PROJECTS_DIR.exists():
        return results

    for project_dir in PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        project_path = project_dir.name
        for jsonl_file in project_dir.glob("*.jsonl"):
            results.append((jsonl_file, project_path))

    return results


def ingest(verbose: bool = True) -> dict:
    """メインのインジェスト処理"""
    if verbose:
        print("モデルをロード中...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    db = get_db()

    ingested = get_ingested_sessions(db)
    if verbose:
        print(f"インデックス済みセッション数: {len(ingested)}")

    jsonl_files = find_jsonl_files()
    if verbose:
        print(f"検出されたjsonlファイル数: {len(jsonl_files)}")

    new_files = 0
    new_chunks = 0
    skipped = 0
    errors = 0

    for filepath, project_path in jsonl_files:
        session_id = filepath.stem
        if session_id in ingested:
            skipped += 1
            continue

        try:
            messages = parse_jsonl(filepath)
            if not messages:
                skipped += 1
                continue

            chunks = create_chunks(messages, session_id, project_path)
            if not chunks:
                skipped += 1
                continue

            # embedding
            texts = [c["text"] for c in chunks]
            prefixed_texts = [f"passage: {t}" for t in texts]
            embeddings = model.encode(prefixed_texts, show_progress_bar=False).tolist()

            # SQLite格納
            for j, c in enumerate(chunks):
                meta = c["metadata"]
                emb = embeddings[j]

                db.execute(
                    "INSERT OR IGNORE INTO chunks (chunk_id, text, session_id, project_path, timestamp, timestamp_epoch, chunk_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (c["id"], c["text"], meta["session_id"], meta["project_path"], meta["timestamp"], meta["timestamp_epoch"], meta["chunk_index"]),
                )
                db.execute(
                    "INSERT OR IGNORE INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                    (c["id"], serialize_f32(emb)),
                )
                db.execute(
                    "INSERT OR IGNORE INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
                    (c["id"], c["text"]),
                )

            db.commit()
            new_files += 1
            new_chunks += len(chunks)

            if verbose and new_files % 20 == 0:
                print(f"  処理済み: {new_files} ファイル, {new_chunks} チャンク")

        except Exception as e:
            errors += 1
            if verbose:
                print(f"  エラー ({filepath.name}): {e}")

    total_in_db = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    db.close()

    result = {
        "new_files": new_files,
        "new_chunks": new_chunks,
        "skipped": skipped,
        "errors": errors,
        "total_in_collection": total_in_db,
    }

    if verbose:
        print(f"\n完了:")
        print(f"  新規ファイル: {new_files}")
        print(f"  新規チャンク: {new_chunks}")
        print(f"  スキップ: {skipped}")
        print(f"  エラー: {errors}")
        print(f"  DB合計: {total_in_db}")

    return result


if __name__ == "__main__":
    ingest()
