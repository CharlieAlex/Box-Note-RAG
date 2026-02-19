# 📒 Box-Note RAG

> 用 AI 搜尋你的筆記 — 基於 LangGraph 的個人知識庫問答系統

Box-Note 是一個個人筆記檔案庫，本專案透過 **RAG（Retrieval-Augmented Generation）** 技術，讓你可以用自然語言向自己的筆記提問，AI 會從筆記中檢索相關內容並生成回答。

## ✨ 功能特色

- 🔍 **智慧檢索** — 將筆記向量化後，透過語意搜尋找到最相關的筆記片段
- 📄 **多格式支援** — 支援 Markdown (`.md`) 與 PDF (`.pdf`) 筆記匯入
- 🤖 **本地 LLM** — 使用 Ollama 運行本地模型，資料完全不外流
- 🔄 **自動優化查詢** — 當檢索結果不佳時，自動改寫問題重新搜尋
- 📝 **對話紀錄** — 自動儲存每次問答紀錄至 JSONL 檔案
- 🎛️ **Prompt 版本管理** — 透過 YAML 管理 Prompt 模板，支援多版本切換

## 🏗️ 系統架構

本專案基於 **LangGraph** 建構一個有狀態的 RAG Pipeline：

```mermaid
graph LR
    A[使用者提問] --> B[Retrieve<br/>向量檢索]
    B --> C[Grade Documents<br/>文件相關性評估]
    C -->|相關性足夠| D[Generate<br/>生成回答]
    C -->|相關性不足| E[Transform Query<br/>改寫問題]
    E --> B
    D --> F[輸出回答]
```

| 節點 | 說明 |
|------|------|
| **Retrieve** | 從 ChromaDB 向量資料庫中檢索 Top-K 相關筆記片段 |
| **Grade Documents** | 使用 LLM 逐一評估文件與問題的相關性 |
| **Transform Query** | 當相關文件不足時，自動改寫問題以優化搜尋 |
| **Generate** | 將篩選後的筆記作為上下文，生成最終回答 |

## 📁 專案結構

```
Box-Note-RAG/
├── app/
│   ├── config.py              # 環境變數與設定 (Pydantic Settings)
│   ├── graph.py               # LangGraph 工作流定義
│   ├── nodes.py               # 各節點邏輯 (retrieve / grade / generate)
│   ├── state.py               # GraphState 型別定義
│   ├── prompts/
│   │   ├── manager.py         # Prompt 版本管理器
│   │   └── templates.yaml     # Prompt 模板 (支援多版本)
│   └── retriever/
│       ├── loaders.py         # 文件載入器 (Markdown / PDF)
│       ├── splitters.py       # 文本切片策略
│       └── vector_store.py    # ChromaDB 向量資料庫
├── scripts/
│   ├── ingest.py              # 筆記匯入腳本
│   └── db_ops.py              # 向量資料庫管理工具
├── data/                      # 向量資料庫與對話紀錄
├── main.py                    # CLI 主程式入口
├── Makefile                   # 常用指令快捷
└── pyproject.toml             # 專案依賴管理 (uv)
```

## 🛠️ 技術棧

| 類別 | 工具 |
|------|------|
| **LLM 框架** | LangChain / LangGraph |
| **本地模型** | Ollama (`llama3.2`) |
| **Embedding 模型** | Ollama (`qwen3-embedding:0.6b`) |
| **向量資料庫** | ChromaDB |
| **套件管理** | uv |
| **設定管理** | Pydantic Settings + `.env` |
| **日誌** | Loguru |
| **Linter** | Ruff |

## 🚀 快速開始

### 前置需求

- Python 3.9
- [uv](https://docs.astral.sh/uv/) 套件管理工具
- [Ollama](https://ollama.ai/) 本地模型運行環境

### 安裝

```bash
# 1. Clone 專案
git clone https://github.com/CharlieAlex/Box-Note-RAG.git
cd Box-Note-RAG

# 2. 安裝依賴
make venv

# 3. 設定環境變數
cp .env.example .env
# 依需求修改 .env 中的模型設定
```

### 下載 Ollama 模型

```bash
# LLM 模型
ollama pull llama3.2

# Embedding 模型
ollama pull qwen3-embedding:0.6b
```

### 匯入筆記

將你的筆記（`.md` / `.pdf`）放到指定資料夾，然後執行匯入：

```bash
make ingest DATA_PATH=/path/to/your/notes
```

### 開始提問

```bash
make run
```

程式會提示你輸入問題，接著自動從筆記中檢索並生成回答。

## 📊 資料庫管理

```bash
# 查看向量資料庫統計
make db-stats

# 直接測試向量檢索（不經過 LLM）
make db-search QUERY="你的搜尋關鍵字"

# 列出已匯入的所有文件來源
make db-sources
```

## ⚙️ 環境變數

| 變數 | 說明 | 預設值 |
|------|------|--------|
| `CHROMA_PATH` | ChromaDB 儲存路徑 | `data/chroma_db` |
| `EMBEDDINGS_MODEL` | Ollama Embedding 模型名稱 | `qwen3-embedding:0.6b` |
| `BATCH_SIZE` | 匯入時的批次大小 | `50` |
| `OLLAMA_MODEL` | Ollama LLM 模型名稱 | `llama3.2` |
