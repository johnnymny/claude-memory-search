# claude-memory-search

Claude Codeの全セッションログをハイブリッド検索できるMCPサーバー。

## 何ができるか

Claude Codeは全セッションの会話を `~/.claude/projects/` にjsonlファイルとして保存している。しかし、セッションを跨いで過去の会話を検索する手段は公式には提供されていない。

このツールはそのjsonlをembeddingしてSQLiteに格納し、MCPサーバーとしてハイブリッド検索（ベクトル検索 + キーワード検索）を提供する。Claude Codeのエージェントが「以前どう解決したか」「過去に同じ問題があったか」を自分で検索できるようになる。

### ユースケース

- 過去に解決した問題の解決策を探す
- 以前の議論や意思決定の経緯を参照する
- MEMORY.mdに書き忘れた知見を過去ログから発掘する

## 構成

```
claude-memory-search/
├── server.py          # MCPサーバー（stdio方式）
├── ingest.py          # セッションログ → SQLite 格納
├── migrate.py         # ChromaDB → SQLite マイグレーション
├── hooks/
│   └── ingest_hook.py # UserPromptSubmit hook（自動インジェスト）
├── requirements.txt
├── data/              # SQLite永続ストレージ（自動生成）
│   └── memory.sqlite3
└── .venv/             # Python仮想環境
```

## 技術スタック

| 項目 | 選定 |
|------|------|
| ベクトル検索 | sqlite-vec（SQLite拡張、ファイルベース） |
| キーワード検索 | SQLite FTS5（trigramトークナイザ） |
| ランキング統合 | RRF（Reciprocal Rank Fusion、k=60） |
| Embedding | `intfloat/multilingual-e5-small`（日本語対応、約100MB） |
| MCPプロトコル | `mcp` Python SDK（FastMCP、stdio方式） |
| Python | 3.12 + venv |

## 検索の仕組み

```
query
  ├─→ sqlite-vec (ベクトル検索: 意味的類似) → ランキング
  ├─→ FTS5 (キーワード検索: 固有名詞・完全一致) → ランキング
  └─→ RRF で統合 → 返却
```

ベクトル検索だけでは固有名詞（ツール名、クラス名、設定値）に弱く、キーワード検索だけでは意味的な類似を見逃す。両方を組み合わせることで、お互いの弱点を補う。

結果には `hit` フィールドがあり、`vec+fts`（両方でヒット）、`vec`（ベクトルのみ）、`fts`（キーワードのみ）のいずれかを示す。

## セットアップ

```bash
# 任意の場所にclone
git clone https://github.com/johnnymny/claude-memory-search.git
cd claude-memory-search

# venv作成 + 依存インストール
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 初回インデックス構築（既存セッションログを取り込む）
.venv/bin/python ingest.py
```

## MCP登録

プロジェクトの `.mcp.json` に追記する。パスはclone先に合わせて書き換えること。

```json
{
  "mcpServers": {
    "memory-search": {
      "command": "<clone先>/claude-memory-search/.venv/bin/python",
      "args": ["<clone先>/claude-memory-search/server.py"]
    }
  }
}
```

例（WSL環境）:
```json
{
  "mcpServers": {
    "memory-search": {
      "command": "/home/user/claude-memory-search/.venv/bin/python",
      "args": ["/home/user/claude-memory-search/server.py"]
    }
  }
}
```

`settings.json` には `mcpServers` フィールドがないため、`.mcp.json` を使う。

## CLAUDE.mdへの追記（推奨）

MCPツールを登録しただけではClaude Codeは自発的に使わない。プロジェクトの `CLAUDE.md` に以下のようなルールを追記することで、エージェントが適切なタイミングで過去ログを検索するようになる。

```markdown
## Memory Search MCP

確信がない時・過去の経験が関係しそうな時は memory_search で過去セッションログを検索せよ。
新しいセッションログを取り込むには memory_ingest を実行する。
```

## MCPツール

### `memory_search(query, limit=5, date_from="", date_to="")`

過去セッションからハイブリッド検索で類似する会話を検索する。

- `query`: 検索クエリ（日本語・英語可）
- `limit`: 返す結果数（デフォルト5、最大20）
- `date_from`: 検索開始日（ISO 8601形式、例: "2026-03-05"）
- `date_to`: 検索終了日（ISO 8601形式、例: "2026-03-06"）
- e5モデルは検索時に `query:` プレフィックス、格納時に `passage:` プレフィックスを使用

### `memory_ingest()`

`~/.claude/projects/` 配下の全jsonlを走査し、未インデックスのファイルのみ処理する（増分更新）。

## インジェスト仕様

- **ソース**: `~/.claude/projects/` 配下の `*.jsonl`
- **チャンク分割**: user/assistantペアを基本単位、4ペアのスライディングウィンドウ（ステップ2）
- **最大チャンク長**: 2,000文字
- **除外**: system, progress, file-history-snapshot, queue-operation
- **tool_use/tool_result**: ツール名と主要入力のみに要約
- **増分更新**: session_id（ファイル名）で既存チェック、処理済みはスキップ
- **格納先**: `data/memory.sqlite3`（chunks + chunks_vec + chunks_fts の3テーブル）

## 自動インジェスト（Hook）

`hooks/ingest_hook.py` をClaude Codeの `UserPromptSubmit` hookに登録すると、セッション開始時に自動でインジェストが走る。手動で `memory_ingest` を呼ぶ必要がなくなる。

`~/.claude/settings.json` に追記:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python -X utf8 <clone先>/claude-memory-search/hooks/ingest_hook.py"
          }
        ]
      }
    ]
  }
}
```

- venvは自動検出（`.venv/` or `venv-win/`）
- 5分間隔のデデュプ付き（短時間に複数セッションを開いても重複実行しない）
- バックグラウンド起動（hookの応答をブロックしない）

## ChromaDBからの移行

以前のバージョン（ChromaDB）からの移行:

```bash
# ChromaDB → SQLite マイグレーション
.venv/bin/pip install sqlite-vec chromadb
.venv/bin/python migrate.py

# 動作確認後、旧データを削除
rm data/chroma.sqlite3
rm -rf data/edbc56a4-*/
```

## 直接実行

```bash
# インジェスト（CLIから直接）
.venv/bin/python ingest.py

# サーバー起動（通常はMCP経由で自動起動）
.venv/bin/python server.py
```

## 注意事項

- **データディレクトリ**: `data/memory.sqlite3`
- **venv必須**: Ubuntu 24.04は externally-managed-environment のため
- **初回起動**: embeddingモデルのダウンロードでstderrにプログレスバーが出る
- **trigram最小長**: FTS5のtrigramトークナイザは3文字未満のクエリトークンを無視する
