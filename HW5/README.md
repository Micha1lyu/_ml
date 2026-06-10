# HW5 - AI Agent with Security Sandbox

基於 v2-agent-xml 擴充，加入**路徑沙箱**與 **LLM 安全審查**的 AI Agent。

## 功能

- 透過 Ollama 呼叫本地 LLM，支援對話記憶與關鍵資訊萃取
- 可執行 shell 命令（`<tool>` 格式）並把結果回饋給 AI
- **安全控管**：命令執行前經過兩道閘口

## 安全機制

### 1. 路徑沙箱（Sandbox）

Agent 預設只能存取 `agent0.py` 所在資料夾及其子目錄。

若命令中含有沙箱以外的路徑，會**攔截並詢問使用者**：

```
⚠️  [安全警告] Agent 嘗試存取沙箱以外的路徑：
   /etc/passwd
沙箱範圍：/home/user/HW5
是否核准此次存取？[y/N]
```

- 輸入 `y` 核准（同樣路徑本次執行不再重複問）
- 輸入 `N`（預設）拒絕，命令回傳 `[BLOCKED]`

### 2. LLM 安全審查（Reviewer）

路徑通過後，再呼叫 `REVIEWER_MODEL` 審查命令是否有潛在危害。

審查原則：
| 允許 | 禁止 |
|---|---|
| 讀檔、ls、cat、grep、find | `rm -rf`、`del /f`、`dd` |
| git、python、node 等開發工具 | `sudo`、`chmod 777` |
| 一般搜尋操作 | curl/wget 下載並執行腳本 |

## 執行

```bash
pip install aiohttp
python agent0.py
```

> 需要本地執行 [Ollama](https://ollama.com/) 並拉好模型：
> ```bash
> ollama pull minimax-m2.5:cloud
> ```

## 設定

在 `agent0.py` 頂部修改：

```python
MODEL = "minimax-m2.5:cloud"      # 主要對話模型
REVIEWER_MODEL = "minimax-m2.5:cloud"  # 安全審查模型（可換成不同 LLM）
MAX_TURNS = 5                       # 保留的對話輪數
```

`WORKSPACE` 自動設定為 `agent0.py` 所在目錄，無需手動修改。

## 版本對照

| 版本 | 說明 |
|---|---|
| v2-agent-xml | 基底版本，用 XML 格式傳遞 prompt，修正 JSON 格式問題 |
| **v3（本版）** | 新增路徑沙箱 + LLM 審查，`pathlib` 跨平台路徑判斷 |

## 檔案結構

```
HW5/
├── agent0.py   # 主程式
└── README.md   # 本文件
```
