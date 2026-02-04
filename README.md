# claude-memory-search

Claude Codeの全セッションログを意味検索できるMCPサーバー。

## 何ができるか

Claude Codeは全セッションの会話を `~/.claude/projects/` にjsonlファイルとして保存している。しかし、セッションを跨いで過去の会話を検索する手段は公式には提供されていない。

このツールはそのjsonlをembeddingしてベクトルDBに格納し、MCPサーバーとして類似検索を提供する。Claude Codeのエージェントが「以前どう解決したか」「過去に同じ問題があったか」を自分で検索できるようになる。

### ユースケース

- 過去に解決した問題の解決策を探す
- 以前の議論や意思決定の経緯を参照する
- MEMORY.mdに書き忘れた知見を過去ログから発掘する

## 構成

```
claude-memory-search/
├── server.py          # MCPサーバー（stdio方式）
├── ingest.py          # セッションログ → ChromaDB 格納
├── requirements.txt
├── data/              # ChromaDB永続ストレージ（自動生成）
│   └── chroma.sqlite3
└── .venv/             # Python仮想環境
```

## 技術スタック

| 項目 | 選定 |
|------|------|
| ベクトルDB | ChromaDB（ファイルベース、サーバー不要） |
| Embedding | `intfloat/multilingual-e5-small`（日本語対応、約100MB） |
| MCPプロトコル | `mcp` Python SDK（FastMCP、stdio方式） |
| Python | 3.12 + venv |

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

## MCPツール

### `memory_search(query, limit=5)`

過去セッションから類似する会話を検索する。

- `query`: 検索クエリ（日本語・英語可）
- `limit`: 返す結果数（デフォルト5、最大20）
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

## 直接実行

```bash
# インジェスト（CLIから直接）
.venv/bin/python ingest.py

# サーバー起動（通常はMCP経由で自動起動）
.venv/bin/python server.py
```

## 注意事項

- **データディレクトリ**: `data/`（`chroma_db/` ではない）
- **venv必須**: Ubuntu 24.04は externally-managed-environment のため
- **初回起動**: embeddingモデルのダウンロード + ウェイトロードでstderrにプログレスバーが出る
- **ChromaDB直接クエリ**: `.query(query_texts=...)` はChromaDBデフォルトのembedding（all-MiniLM-L6-v2）を使うため、e5-smallでインジェストしたデータとモデル不一致になる。必ず `query_embeddings` でe5モデルのembeddingを渡すこと
