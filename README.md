# rag-hands-on

RAGハンズオン用のリポジトリ。Neo4j（グラフデータベース）と LangChain、Google Gemini API を組み合わせた **Graph RAG** の実装をJupyter Notebook上で学べます。

## 概要

このリポジトリでは、以下の内容をハンズオン形式で扱います。

- Neo4j上でのナレッジグラフの手動作成・クエリ実行
- LangChain (`langchain-neo4j`) と Gemini モデルを使った自然言語 → Cypherクエリ変換
- グラフの実行結果を踏まえた自然言語での回答生成（Graph RAGチェイン）
- `LLMGraphTransformer` によるテキストからのナレッジグラフ自動生成

## ディレクトリ構成

```
.
├── README.md
├── LICENSE
├── pyproject.toml        # プロジェクト定義・依存パッケージ
├── uv.lock                # uvによる依存関係ロックファイル
├── .python-version         # 使用するPythonバージョン (3.13)
├── .env.sample             # 環境変数のサンプル
├── img/                     # ノートブック内で参照する画像
│   ├── sar_graph.png
│   └── sar_profile.png
└── notebook/
    └── graph_rag/
        └── graph_rag.ipynb  # Graph RAGハンズオン用ノートブック
```

## セットアップ手順

### 1. 前提条件

- Python 3.13 以上
- [uv](https://docs.astral.sh/uv/) がインストールされていること
- Neo4jデータベース（Neo4j AuraDB もしくはローカル/Dockerでの起動）
- Google Gemini APIキー

### 2. リポジトリの取得と依存関係の同期

```bash
git clone <このリポジトリのURL>
cd rag-hands-on

# pyproject.toml / uv.lock に基づいて依存関係を同期
uv sync
```

`uv sync` により、`.python-version` で指定された Python 3.13 環境と、`langchain` / `langchain-neo4j` / `langchain-google-genai` / `jupyterlab` などの依存パッケージが自動的にセットアップされます。

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

### 4. Neo4jの登録

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

起動後、`notebook/graph_rag/graph_rag.ipynb` を開いてセルを上から順に実行してください。
