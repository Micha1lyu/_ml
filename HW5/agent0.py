#!/usr/bin/env python3
# agent0.py - AI Agent with memory, tool feedback, and security sandbox
# Run: python agent0.py

import subprocess
import os
import asyncio
import aiohttp
import re
from pathlib import Path

# ─── Configuration ───

WORKSPACE = os.path.abspath(os.path.dirname(__file__))  # 沙箱 = 本檔案所在資料夾
MODEL = "minimax-m2.5:cloud"
REVIEWER_MODEL = "minimax-m2.5:cloud"  # 可換成另一個 LLM
MAX_TURNS = 5

# ─── Memory ───

conversation_history = []
key_info = []
outside_access_granted: set[str] = set()  # 已經授權過的路徑（不再重複問）

# ─── Ollama API ───

async def call_ollama(prompt: str, system: str = "") -> str:
    """Call Ollama API"""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    payload = {
        "model": MODEL,
        "prompt": full_prompt,
        "stream": False
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            result = await resp.json()
            return result.get("response", "").strip()

# ─── Security: LLM Reviewer ───

async def review_command(cmd: str) -> tuple[bool, str]:
    """用另一個 LLM 審查命令是否安全"""
    review_prompt = f"""你是安全審查者。請判斷以下 shell 命令是否安全可以執行。

安全原則：
1. 允許讀取檔案、瀏覽目錄、搜尋程式碼
2. 允許執行無害的開發工具（git, ls, cat, grep, find, python, node 等）
3. 禁止會刪除資料的命令（rm -rf, del /f, dd, mkfs 等）
4. 禁止會修改系統的命令（sudo, chmod 777, 修改登錄檔等）
5. 禁止網路相關的危險操作（curl/wget 下載並執行腳本等）
6. 禁止任何可能造成資料洩露或系統傷害的命令

要審查的命令：
{cmd}

請嚴格按照以下格式輸出：
<review>
  <verdict>SAFE</verdict>
</review>
或
<review>
  <verdict>UNSAFE</verdict>
  <reason>原因說明</reason>
</review>

不要輸出其他內容。"""

    payload = {
        "model": REVIEWER_MODEL,
        "prompt": review_prompt,
        "stream": False
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                result = await resp.json()
                response = result.get("response", "").strip()

        verdict_match = re.search(r"<verdict>(.*?)</verdict>", response, re.DOTALL)
        reason_match = re.search(r"<reason>(.*?)</reason>", response, re.DOTALL)

        if verdict_match and verdict_match.group(1).strip().upper() == "SAFE":
            return True, ""
        else:
            reason = reason_match.group(1).strip() if reason_match else response
            return False, reason
    except Exception as e:
        return False, f"審查失敗: {e}"

# ─── Security: Sandbox Path Check ───

def _extract_paths_from_cmd(cmd: str) -> list[str]:
    """從 shell 命令中萃取所有路徑"""
    paths = []
    # Unix 絕對路徑
    for m in re.finditer(r'(?:^|\s)(/[^\s;|&>]+)', cmd):
        paths.append(m.group(1))
    # Windows 絕對路徑 (C:\... 或 C:/...)
    for m in re.finditer(r'(?:^|\s)([A-Za-z]:[/\\][^\s;|&>]*)', cmd):
        paths.append(m.group(1))
    # 相對 ../
    for m in re.finditer(r'(?:^|\s)(\.\.(?:[/\\][^\s;|&>]*)?)', cmd):
        paths.append(m.group(1))
    # ~ 開頭
    for m in re.finditer(r'(?:^|\s)(~[/\\][^\s;|&>]*)', cmd):
        paths.append(m.group(1))
    return paths


def check_outside_access(cmd: str) -> tuple[bool, list[str]]:
    """
    回傳 (has_outside, [suspicious_paths])
    has_outside = True 代表命令嘗試存取 WORKSPACE 以外的路徑
    """
    sandbox = Path(WORKSPACE).resolve()
    suspects = []

    raw_paths = _extract_paths_from_cmd(cmd)
    for p in raw_paths:
        try:
            resolved = Path(os.path.expanduser(p)).resolve()
            # 使用 is_relative_to 確認是否在沙箱內（Python 3.9+）
            if not resolved.is_relative_to(sandbox):
                suspects.append(str(resolved))
        except Exception:
            pass  # 解析失敗的路徑跳過

    return len(suspects) > 0, suspects


def ask_outside_access(suspects: list[str]) -> bool:
    """
    顯示警告，詢問使用者是否核准沙箱外存取
    回傳 True = 核准，False = 拒絕
    """
    print("\n" + "=" * 60)
    print("⚠️  [安全警告] Agent 嘗試存取沙箱以外的路徑：")
    for p in suspects:
        print(f"   {p}")
    print(f"沙箱範圍：{WORKSPACE}")
    print("=" * 60)
    while True:
        ans = input("是否核准此次存取？[y/N] ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", ""):
            return False
        print("請輸入 y 或 n")


# ─── Security Gate ───

async def security_gate(cmd: str) -> tuple[bool, str]:
    """
    執行命令前的安全閘口：
    1. 路徑沙箱檢查 → 若存取外部，詢問使用者
    2. LLM reviewer 審查命令安全性
    回傳 (allowed, reason)
    """
    # 1. 沙箱路徑檢查
    has_outside, suspects = check_outside_access(cmd)
    if has_outside:
        key = ",".join(sorted(suspects))
        if key not in outside_access_granted:
            approved = ask_outside_access(suspects)
            if not approved:
                return False, f"使用者拒絕存取沙箱外路徑：{suspects}"
            outside_access_granted.add(key)

    # 2. LLM 安全審查
    safe, reason = await review_command(cmd)
    if not safe:
        print(f"\n🚫 [LLM 審查] 命令被拒絕：{reason}")
        return False, reason

    return True, ""

# ─── Memory Management ───

def build_context():
    context_parts = []
    if key_info:
        items_xml = "\n".join(f"  <item>{k}</item>" for k in key_info)
        context_parts.append(f"<memory>\n{items_xml}\n</memory>")
    if conversation_history:
        context_parts.append("<history>\n" + "\n".join(conversation_history[-MAX_TURNS * 2:]) + "\n</history>")
    return "\n\n".join(context_parts)


def update_memory(user_input, assistant_response, tool_result=None):
    conversation_history.append(f"  <user>{user_input}</user>")
    conversation_history.append(f"  <assistant>{assistant_response}</assistant>")
    if tool_result:
        conversation_history.append(f"  <tool>{tool_result[:500]}</tool>")

    while len(conversation_history) > MAX_TURNS * 4:
        conversation_history.pop(0)


async def extract_key_info(user_input, assistant_response):
    extract_prompt = f"""根據這段對話，有沒有需要長期記憶的關鍵資訊？
如果有，用以下格式輸出（最多 2 項）。如果沒有，輸出 <memory></memory>。

<memory>
  <item>要記憶的資訊 1</item>
  <item>要記憶的資訊 2</item>
</memory>

對話：
<user>{user_input}</user>
<assistant>{assistant_response}</assistant>"""

    try:
        response = await call_ollama(extract_prompt)
        items = re.findall(r"<item>(.*?)</item>", response, re.DOTALL)
        for item in items:
            item = item.strip()
            if item and item not in key_info:
                key_info.append(item)
                if len(key_info) > 10:
                    key_info.pop(0)
    except Exception:
        pass

# ─── Tool Execution ───

async def execute_tool(cmd: str) -> str:
    """執行 shell 命令前先過安全閘口"""
    allowed, reason = await security_gate(cmd)
    if not allowed:
        return f"[BLOCKED] {reason}"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=WORKSPACE
        )
        output = result.stdout or result.stderr
        return output[:2000] if output else "(無輸出)"
    except subprocess.TimeoutExpired:
        return "[錯誤] 命令逾時"
    except Exception as e:
        return f"[錯誤] {e}"

# ─── Agent Core ───

SYSTEM_PROMPT = f"""你是一個 AI 助理，可以執行 shell 命令來完成任務。
工作目錄（沙箱）：{WORKSPACE}

工具使用規則：
- 需要執行命令時，用以下格式包住命令：
  <tool>命令</tool>
- 一次只能執行一個命令
- 命令執行後你會收到輸出，再根據結果繼續回答

安全提醒：存取沙箱以外的路徑需要使用者核准。"""


async def agent_turn(user_input: str) -> str:
    context = build_context()

    prompt = f"""{context}

<user>{user_input}</user>

請根據以上資訊回答。如果需要執行命令，請使用 <tool>命令</tool> 格式。"""

    response = await call_ollama(prompt, system=SYSTEM_PROMPT)

    # 解析並執行 tool calls
    tool_match = re.search(r"<tool>(.*?)</tool>", response, re.DOTALL)
    tool_result = None

    if tool_match:
        cmd = tool_match.group(1).strip()
        print(f"\n🔧 執行命令：{cmd}")
        tool_result = await execute_tool(cmd)
        print(f"📤 輸出：{tool_result[:200]}{'...' if len(tool_result) > 200 else ''}")

        # 把 tool 結果餵回去讓 agent 繼續
        followup_prompt = f"""{context}

<user>{user_input}</user>
<assistant>{response}</assistant>
<tool_result>{tool_result}</tool_result>

請根據命令輸出，給出最終回答。"""
        response = await call_ollama(followup_prompt, system=SYSTEM_PROMPT)

    update_memory(user_input, response, tool_result)
    asyncio.create_task(extract_key_info(user_input, response))

    return response


async def main():
    print("=" * 60)
    print(f"🤖 Agent0 啟動")
    print(f"📁 沙箱目錄：{WORKSPACE}")
    print(f"🔒 安全模式：啟用（路徑沙箱 + LLM 審查）")
    print("輸入 'exit' 結束")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("再見！")
            break

        print("🤔 思考中...")
        try:
            response = await agent_turn(user_input)
            print(f"\nAgent：{response}")
        except Exception as e:
            print(f"\n[錯誤] {e}")


if __name__ == "__main__":
    asyncio.run(main())
