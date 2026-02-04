"""
server.py - Claude Memory Search MCP Server

過去のClaude Codeセッションログを類似検索するMCPサーバー。
stdio方式で動作し、Claude Codeから直接呼び出せる。
"""

import json
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import (
    DATA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    get_chroma_client,
    get_collection,
    ingest,
)

# --- グローバル初期化（起動時に1回だけ） ---
_model = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = get_collection(client)
    return _collection


# --- MCPサーバー定義 ---
mcp = FastMCP("memory-search")


@mcp.tool()
def memory_search(query: str, limit: int = 5) -> str:
    """過去のClaude Codeセッションログから類似する会話を検索する。

    確信がない時、過去に同じ問題を解決した記憶がありそうな時、
    以前の議論や決定事項を参照したい時に使う。

    Args:
        query: 検索クエリ（日本語・英語どちらでも可）
        limit: 返す結果の最大数（デフォルト: 5）
    """
    model = _get_model()
    collection = _get_collection()

    if collection.count() == 0:
        return json.dumps({
            "error": "インデックスが空です。先に memory_ingest を実行してください。",
            "results": [],
        }, ensure_ascii=False)

    # e5モデルは検索時に "query: " プレフィックスを付ける
    query_embedding = model.encode(f"query: {query}", show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(limit, 20),
        include=["documents", "metadatas", "distances"],
    )

    formatted = []
    for i in range(len(results["ids"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # cosine距離 → 類似度スコアに変換
        similarity = 1.0 - distance

        # project_pathから読みやすい名前に
        project = meta.get("project_path", "unknown")
        # "-mnt-c-Users-xxx-" 形式のプレフィックスを除去
        project = re.sub(r"^-mnt-c-Users-[^-]+-", "", project)
        project = project.replace("-", "/")

        # タイムスタンプをフォーマット
        timestamp = meta.get("timestamp", "")

        formatted.append({
            "score": round(similarity, 3),
            "project": project,
            "session_id": meta.get("session_id", "")[:8] + "...",
            "timestamp": timestamp,
            "conversation": doc[:1500],
        })

    return json.dumps({
        "query": query,
        "total_indexed": collection.count(),
        "results": formatted,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def memory_ingest() -> str:
    """Claude Codeのセッションログをインデックスに取り込む。

    ~/.claude/projects/ 配下の全jsonlファイルを走査し、
    未インデックスのファイルのみ処理する（増分更新）。
    初回実行時はモデルのダウンロードも含めて時間がかかる場合がある。
    """
    result = ingest(verbose=False)
    return json.dumps({
        "status": "完了",
        "new_files": result["new_files"],
        "new_chunks": result["new_chunks"],
        "skipped_files": result["skipped"],
        "errors": result["errors"],
        "total_chunks_in_index": result["total_in_collection"],
    }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
