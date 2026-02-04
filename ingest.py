"""
ingest.py - Claude Code セッションログを ChromaDB に格納する

jsonlファイルをパースし、user/assistantペアのスライディングウィンドウで
チャンク分割、multilingual-e5-small でembeddingして ChromaDB に保存する。
増分更新対応: 既にインデックス済みのファイルはスキップする。
"""

import json
import os
import re
import hashlib
from pathlib import Path
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer

# --- 設定 ---
PROJECTS_DIR = Path.home() / ".claude" / "projects"
DATA_DIR = Path(__file__).parent / "data"
COLLECTION_NAME = "claude_sessions"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
WINDOW_SIZE = 4  # user/assistantペア数
WINDOW_STEP = 2  # スライドステップ
MAX_CHUNK_CHARS = 2000  # チャンクの最大文字数

# 除外するメッセージtype
SKIP_TYPES = {"system", "progress", "file-history-snapshot", "queue-operation"}


def get_chroma_client() -> chromadb.ClientAPI:
    """ChromaDB永続クライアントを返す"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(DATA_DIR))


def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """コレクションを取得または作成する"""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_ingested_sessions(collection: chromadb.Collection) -> set[str]:
    """既にインデックス済みのsession_idを取得する"""
    try:
        result = collection.get(include=["metadatas"])
        if result["metadatas"]:
            return {m["session_id"] for m in result["metadatas"] if "session_id" in m}
    except Exception:
        pass
    return set()


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
                # ツール入力の要約: 主要なキーだけ
                summary_parts = []
                if isinstance(tool_input, dict):
                    for key in ["command", "query", "pattern", "file_path", "prompt", "url"]:
                        if key in tool_input:
                            val = str(tool_input[key])[:100]
                            summary_parts.append(f"{key}={val}")
                input_summary = ", ".join(summary_parts) if summary_parts else ""
                parts.append(f"[tool: {tool_name}({input_summary})]")

            elif item_type == "tool_result":
                # tool_resultの内容は簡略化
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
    session_id = filepath.stem  # UUIDファイル名がsession_id

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

            # 除外するtype
            if msg_type in SKIP_TYPES:
                continue

            # user or assistant メッセージのみ
            if msg_type not in ("user", "assistant"):
                continue

            # メッセージ本体の取得
            role = msg_type
            content = None

            # 新形式: 直接 contentフィールド
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
    # user/assistantペアに分割
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
            # assistantが先に来た場合はスキップ
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

        # チャンクテキストを構築
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

        # 最初のペアのタイムスタンプを使用
        timestamp = window[0][0].get("timestamp", "")

        # 一意IDを生成
        chunk_id = hashlib.md5(
            f"{session_id}:{chunk_index}".encode()
        ).hexdigest()

        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "session_id": session_id,
                "project_path": project_path,
                "timestamp": str(timestamp),
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

    client = get_chroma_client()
    collection = get_collection(client)

    # 既にインデックス済みのセッションを取得
    ingested = get_ingested_sessions(collection)
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

            # embedding + ChromaDB格納（バッチ処理）
            texts = [c["text"] for c in chunks]
            # multilingual-e5-small は "query: " プレフィックスが推奨
            prefixed_texts = [f"passage: {t}" for t in texts]
            embeddings = model.encode(prefixed_texts, show_progress_bar=False).tolist()

            ids = [c["id"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            new_files += 1
            new_chunks += len(chunks)

            if verbose and new_files % 20 == 0:
                print(f"  処理済み: {new_files} ファイル, {new_chunks} チャンク")

        except Exception as e:
            errors += 1
            if verbose:
                print(f"  エラー ({filepath.name}): {e}")

    result = {
        "new_files": new_files,
        "new_chunks": new_chunks,
        "skipped": skipped,
        "errors": errors,
        "total_in_collection": collection.count(),
    }

    if verbose:
        print(f"\n完了:")
        print(f"  新規ファイル: {new_files}")
        print(f"  新規チャンク: {new_chunks}")
        print(f"  スキップ: {skipped}")
        print(f"  エラー: {errors}")
        print(f"  コレクション合計: {collection.count()}")

    return result


if __name__ == "__main__":
    ingest()
