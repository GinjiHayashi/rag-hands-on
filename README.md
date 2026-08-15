# rag-hands-on

RAGハンズオン用のリポジトリ。LangChain と Google Gemini API を使い、**Naive RAG**（ベクトルDB検索によるRAG）と **Graph RAG**（Neo4jのグラフDBを使ったRAG）の2種類の実装をJupyter Notebook上で学べます。

## 概要

このリポジトリでは、以下の内容をハンズオン形式で扱います。

### Naive RAG (`notebook/naive_rag/`)

- PDF文書の読み込みとベクトルデータベース（Chroma）への格納
- Gemini埋め込みモデルによるベクトル化と類似度検索（Retriever）
- 検索結果をプロンプトに埋め込んで回答させるRAGチェインの構築
- RAGあり／なしでのLLM回答の比較

### Graph RAG (`notebook/graph_rag/`)

- Neo4j上でのナレッジグラフの手動作成・クエリ実行
- LangChain (`langchain-neo4j`) と Gemini モデルを使った自然言語 → Cypherクエリ変換
- グラフの実行結果を踏まえた自然言語での回答生成（Graph RAGチェイン）
- `LLMGraphTransformer` によるテキストからのナレッジグラフ自動生成

各ノートブックのコード解説は、それぞれの `code_overview.md` にまとめています（初学者向けに、LangChain・Neo4j・埋め込み等の基礎概念から解説しています）。

- [notebook/naive_rag/code_overview.md](notebook/naive_rag/code_overview.md)
- [notebook/graph_rag/code_overview.md](notebook/graph_rag/code_overview.md)

## ディレクトリ構成

```
.
├── README.md
├── LICENSE
├── pyproject.toml            # プロジェクト定義・依存パッケージ
├── uv.lock                    # uvによる依存関係ロックファイル
├── .python-version             # 使用するPythonバージョン (3.13)
├── .env.sample                 # 環境変数のサンプル
├── docs/                        # Naive RAGで検索対象とするPDF文書
│   ├── employee_training_manual_faq.pdf
│   └── internal_guidelines_for_it_devices_and_ai_use.pdf
├── img/                          # ノートブック内で参照する画像
│   ├── sar_graph.png
│   └── sar_profile.png
└── notebook/
    ├── naive_rag/
    │   ├── naive_rag.ipynb        # Naive RAGハンズオン用ノートブック
    │   └── code_overview.md        # naive_rag.ipynbのコード解説
    └── graph_rag/
        ├── graph_rag.ipynb        # Graph RAGハンズオン用ノートブック
        └── code_overview.md        # graph_rag.ipynbのコード解説
```

## セットアップ手順

### 1. 前提条件

- Python 3.13 以上
- [uv](https://docs.astral.sh/uv/) がインストールされていること
- Google Gemini APIキー（Naive RAG / Graph RAG 共通で必要）
- Neo4jデータベース（Neo4j AuraDB もしくはローカル/Dockerでの起動）※ Graph RAGのみ必要

### 2. リポジトリの取得と依存関係の同期

```bash
git clone <このリポジトリのURL>
cd rag-hands-on

# pyproject.toml / uv.lock に基づいて依存関係を同期
uv sync
```

`uv sync` により、`.python-version` で指定された Python 3.13 環境と、`langchain` / `langchain-neo4j` / `langchain-chroma` / `langchain-google-genai` / `jupyterlab` などの依存パッケージが自動的にセットアップされます。

### 3. 環境変数の設定

`.env.sample` をコピーして `.env` を作成し、各値を設定します。

```bash
cp .env.sample .env
```

`.env` の中身：

```
# Gemini APIキーの設定
GEMINI_API_KEY=

# Neo4j 認証情報
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
```

### 4. Neo4jの登録（Graph RAGのみ必要）

`notebook/naive_rag/` はNeo4jを使用しないため、Naive RAGのみ試す場合はこの手順はスキップして構いません。

Neo4j AuraDB（マネージドサービス）を利用する場合の手順例です。

1. [Neo4j Aura](https://neo4j.com/product/auradb/) にアクセスし、アカウントを作成する
2. 無料の `AuraDB Free` インスタンスを新規作成する
3. インスタンス作成時に表示される以下の情報を控えておく
   - **接続URI**（例: `neo4j+s://xxxxxxxx.databases.neo4j.io`）
   - **ユーザー名**（デフォルトは `neo4j`）
   - **パスワード**
4. 控えた情報を `.env` の各項目に設定する

   ```
   NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=<発行されたパスワード>
   ```

ローカル環境やDockerでNeo4jを起動する場合は、起動したインスタンスの接続情報（デフォルトは `bolt://localhost:7687`、ユーザー名 `neo4j`）を同様に設定してください。

### 5. Gemini APIの登録

1. [Google AI Studio](https://aistudio.google.com/) にアクセスし、Googleアカウントでログインする
2. 「Get API key」からAPIキーを新規発行する
3. 発行されたAPIキーを `.env` の `GEMINI_API_KEY` に設定する

   ```
   GEMINI_API_KEY=<発行されたAPIキー>
   ```

### 6. ノートブックの起動

```bash
uv run jupyter lab
```

起動後、以下のいずれかのノートブックを開いてセルを上から順に実行してください。

- `notebook/naive_rag/naive_rag.ipynb`（Neo4jのセットアップ不要、Gemini APIキーのみでOK）
- `notebook/graph_rag/graph_rag.ipynb`（Neo4j / Gemini APIキーの両方が必要）

各ノートブックのコード解説は、対応する `code_overview.md` を参照してください。
