"""
server.py - Claude Memory Search MCP Server

過去のClaude Codeセッションログをハイブリッド検索するMCPサーバー。
sqlite-vec (ベクトル検索) + FTS5 (キーワード検索) → RRF統合。
stdio方式で動作し、Claude Codeから直接呼び出せる。
"""

import json
import re
import sqlite3
import struct
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
import sqlite_vec

from ingest import (
    DATA_DIR,
    SQLITE_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    get_db,
    ingest,
)

# --- RRF定数 ---
RRF_K = 60  # Reciprocal Rank Fusion の k パラメータ

# --- グローバル初期化（起動時に1回だけ） ---
_model = None
_db = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        _db = get_db()
    return _db


def serialize_f32(vec: list[float]) -> bytes:
    """float list → little-endian bytes for sqlite-vec"""
    return struct.pack(f"<{len(vec)}f", *vec)


def _vector_search(db: sqlite3.Connection, query_embedding: list[float], limit: int, epoch_from: float | None, epoch_to: float | None) -> list[tuple[str, int]]:
    """ベクトル検索 → [(chunk_id, rank), ...]"""
    # sqlite-vec KNN query
    # 日付フィルタはベクトル検索後にchunksテーブルと結合して適用
    fetch_limit = limit * 3  # フィルタで減る分を多めに取る

    rows = db.execute(
        """
        SELECT v.chunk_id, v.distance
        FROM chunks_vec v
        WHERE v.embedding MATCH ?
        ORDER BY v.distance
        LIMIT ?
        """,
        (serialize_f32(query_embedding), fetch_limit),
    ).fetchall()

    # 日付フィルタ適用
    if epoch_from or epoch_to:
        filtered = []
        for chunk_id, dist in rows:
            row = db.execute(
                "SELECT timestamp_epoch FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                continue
            ts = row[0]
            if epoch_from and ts < epoch_from:
                continue
            if epoch_to and ts >= epoch_to:
                continue
            filtered.append((chunk_id, dist))
        rows = filtered

    return [(chunk_id, rank + 1) for rank, (chunk_id, _) in enumerate(rows[:limit * 2])]


def _fts_search(db: sqlite3.Connection, query: str, limit: int, epoch_from: float | None, epoch_to: float | None) -> list[tuple[str, int]]:
    """FTS5キーワード検索 → [(chunk_id, rank), ...]"""
    # trigramトークナイザではクエリの各トークンをダブルクォートで囲んで
    # FTS5 boolean構文の誤解釈を防ぐ。OR結合で部分一致を拾う
    tokens = query.split()
    escaped = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens if len(t) >= 3)

    if not escaped:
        return []

    if epoch_from or epoch_to:
        # FTS結果をchunksテーブルと結合して日付フィルタ
        conditions = []
        params = [escaped]
        if epoch_from:
            conditions.append("c.timestamp_epoch >= ?")
            params.append(epoch_from)
        if epoch_to:
            conditions.append("c.timestamp_epoch < ?")
            params.append(epoch_to)
        where_clause = " AND ".join(conditions)
        params.append(limit * 2)

        rows = db.execute(
            f"""
            SELECT f.chunk_id, rank
            FROM chunks_fts f
            JOIN chunks c ON c.chunk_id = f.chunk_id
            WHERE chunks_fts MATCH ? AND {where_clause}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT chunk_id, rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (escaped, limit * 2),
        ).fetchall()

    return [(chunk_id, rank_pos + 1) for rank_pos, (chunk_id, _) in enumerate(rows)]


def _rrf_merge(vec_results: list[tuple[str, int]], fts_results: list[tuple[str, int]], limit: int) -> list[str]:
    """RRF (Reciprocal Rank Fusion) で2つのランキングを統合"""
    scores: dict[str, float] = {}

    for chunk_id, rank in vec_results:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    for chunk_id, rank in fts_results:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return sorted_ids[:limit]


# --- MCPサーバー定義 ---
mcp = FastMCP("memory-search")


@mcp.tool()
def memory_search(query: str, limit: int = 5, date_from: str = "", date_to: str = "") -> str:
    """過去のClaude Codeセッションログから類似する会話を検索する。

    ベクトル検索（意味的類似）とキーワード検索（固有名詞・完全一致）を
    組み合わせたハイブリッド検索。

    確信がない時、過去に同じ問題を解決した記憶がありそうな時、
    以前の議論や決定事項を参照したい時に使う。
    日付フィルタで期間を絞り込める。「3日前のXXについて」のような
    時系列検索にはdate_from/date_toを使う。

    Args:
        query: 検索クエリ（日本語・英語どちらでも可）
        limit: 返す結果の最大数（デフォルト: 5）
        date_from: 検索開始日（ISO 8601形式、例: "2026-03-05"）。この日以降の会話に絞る
        date_to: 検索終了日（ISO 8601形式、例: "2026-03-06"）。この日より前の会話に絞る
    """
    model = _get_model()
    db = _get_db()

    total = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if total == 0:
        return json.dumps({
            "error": "インデックスが空です。先に memory_ingest を実行してください。",
            "results": [],
        }, ensure_ascii=False)

    # 日付フィルタ
    from datetime import datetime as _dt, timezone as _tz
    epoch_from = epoch_to = None
    try:
        epoch_from = _dt.fromisoformat(date_from).replace(tzinfo=_tz.utc).timestamp() if date_from else None
        epoch_to = _dt.fromisoformat(date_to).replace(tzinfo=_tz.utc).timestamp() if date_to else None
    except Exception:
        pass

    # ベクトル検索
    query_embedding = model.encode(f"query: {query}", show_progress_bar=False).tolist()
    vec_results = _vector_search(db, query_embedding, limit, epoch_from, epoch_to)

    # FTS5キーワード検索
    fts_results = _fts_search(db, query, limit, epoch_from, epoch_to)

    # RRF統合
    merged_ids = _rrf_merge(vec_results, fts_results, min(limit, 20))

    # 結果取得
    formatted = []
    # RRFスコアを保持
    rrf_scores: dict[str, float] = {}
    for chunk_id, rank in vec_results:
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    for chunk_id, rank in fts_results:
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    # どちらの検索でヒットしたかを記録
    vec_ids = {cid for cid, _ in vec_results}
    fts_ids = {cid for cid, _ in fts_results}

    for chunk_id in merged_ids:
        row = db.execute(
            "SELECT text, session_id, project_path, timestamp FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            continue

        text, session_id, project_path, timestamp = row

        project = project_path
        project = re.sub(r"^-mnt-c-Users-[^-]+-", "", project)
        project = project.replace("-", "/")

        hit_sources = []
        if chunk_id in vec_ids:
            hit_sources.append("vec")
        if chunk_id in fts_ids:
            hit_sources.append("fts")

        formatted.append({
            "score": round(rrf_scores.get(chunk_id, 0.0), 4),
            "hit": "+".join(hit_sources),
            "project": project,
            "session_id": session_id[:8] + "...",
            "timestamp": timestamp,
            "conversation": text[:1500],
        })

    return json.dumps({
        "query": query,
        "total_indexed": total,
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
