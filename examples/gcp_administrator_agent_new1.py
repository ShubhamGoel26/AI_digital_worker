#!/usr/bin/env python
"""
Enhanced GCP automation script with dependency handling
--------------------------------
Modifies the AdministratorAgent to:

1. Load tasks from tasks.xlsx and segregate into independent and dependent tasks.
2. Runs a first “login” agent for manual GCP sign-in.
3. For independent tasks: navigates to Google Cloud Console before execution.
4. For dependent tasks: runs sequentially without navigation between tasks.
5. Saves per-task logs/screenshots and generates a summary report.

Requires: pandas, openpyxl
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from dotenv import load_dotenv
from mss import mss
from PIL import Image

import vertexai
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI
from browser_use import Agent

# ---------------------------------------------------------------------------
# 🛠️ Environment & directory setup
# ---------------------------------------------------------------------------

load_dotenv()
BASE_DIR = Path("runs\Gemini")
BASE_DIR.mkdir(exist_ok=True)

vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

# ---------------------------------------------------------------------------
# 📝 Utility callbacks
# ---------------------------------------------------------------------------

def serialize_history(history: Any) -> Any:
    """Best-effort JSON serialization of LangChain run history."""
    try:
        out: list[Any] = []
        for item in history:
            if hasattr(item, "dict"):
                out.append(item.dict)
            elif isinstance(item, (dict, list, str, int, float, bool, type(None))):
                out.append(item)
            else:
                out.append(str(item))
        return out
    except Exception as e:
        return {"error": f"Failed to serialize history: {e}"}


class LLMLogCallback(BaseCallbackHandler):
    """Dumps prompts & completions to a log file."""
    def __init__(self, file: Path):
        self.file = file

    def _w(self, text: str) -> None:
        with self.file.open("a", encoding="utf-8") as f:
            f.write(text)
            f.flush()

    def on_llm_start(self, serialized, prompts, **kwargs):
        model_name = serialized.get("kwargs", {}).get("model_name", "unknown")
        self._w(f"\n=== LLM Prompt (model={model_name}) ===\n")
        for p in prompts:
            self._w(f"{p}\n")

    def on_llm_end(self, response, **kwargs):
        self._w("\n=== LLM Response ===\n")
        for gens in response.generations:
            for gen in gens:
                try:
                    self._w(json.dumps(json.loads(gen.text), indent=4, ensure_ascii=False) + "\n")
                except json.JSONDecodeError:
                    self._w(gen.text + "\n")
        self._w("\n")

    def on_llm_error(self, error, **kwargs):
        self._w(f"\n=== LLM ERROR ===\n{error}\n\n")


class ScreenshotHandler(logging.Handler):
    """Captures partial-screen PNGs when browser-use emits key log lines."""
    TRIGGERS = [
        "Action 1/1:",
        "Next goal:",
        "Searched for",
        "Navigating to",
        "Entering",
        "Clicking",
        "User logged in",
        "Clicked",
    ]

    def __init__(self, json_log: Path, img_dir: Path):
        super().__init__()
        self.json_log = json_log
        self.img_dir = img_dir
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.step = 1
        self.last_goal = ""
        self.debug_file = img_dir / "debug_log.txt"

    async def _snap(self, fname: Path):
        await asyncio.sleep(3)
        try:
            with mss() as sct:
                mon = {"top": 100, "left": 100, "width": 1200, "height": 800}
                shot = sct.grab(mon)
                Image.frombytes("RGB", (shot.width, shot.height), shot.rgb).save(fname, "PNG")
        except Exception as e:
            print(f"Screenshot error: {e}")

    def emit(self, record):
        msg = record.getMessage()
        with self.debug_file.open("a", encoding="utf-8") as f:
            f.write(f"DEBUG: {msg}\n")

        if any(t in msg for t in self.TRIGGERS):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = self.img_dir / f"step_{self.step}_{ts}.png"
            asyncio.create_task(self._snap(fname))

            entry = {
                "step": self.step,
                "action": msg,
                "next_goal": self.last_goal,
                "screenshot": str(fname),
            }
            with self.json_log.open("a", encoding="utf-8") as f:
                json.dump(entry, f)
                f.write("\n")

            if "Next goal:" in msg:
                self.last_goal = msg
            print(f"Step {self.step}: captured {fname}")
            self.step += 1

# ---------------------------------------------------------------------------
# 📋 Task definition and loading
# ---------------------------------------------------------------------------

@dataclass
class Task:
    number: int
    description: str
    prerequisite: Optional[int] = None

def load_tasks(file_path: Path) -> List[Task]:
    """Loads tasks from tasks.xlsx, parsing dependencies."""
    df = pd.read_excel(file_path)
    tasks = []
    for _, row in df.iterrows():
        number = row['Task Number']
        description = row['Task Description']
        dep = row['Dependency/Prerequisite']
        prereq = None
        if pd.notna(dep):
            match = re.search(r'Requires step (\d+)', dep)
            if match:
                prereq = int(match.group(1))
        tasks.append(Task(number=number, description=description, prerequisite=prereq))
    return tasks

# ---------------------------------------------------------------------------
# ⌛ TaskReport for summary
# ---------------------------------------------------------------------------

@dataclass
class TaskReport:
    number: int
    task: str
    result: str
    status: str  # "completed" or "failed"

# ---------------------------------------------------------------------------
# 🤖 AdministratorAgent
# ---------------------------------------------------------------------------

class AdministratorAgent:
    """Runs tasks sequentially, handling independent and dependent tasks differently."""
    def __init__(self, tasks: List[Task], browser, browser_context, base_dir: Path):
        self.tasks = tasks
        self.browser = browser
        self.browser_context = browser_context
        self.base_dir = base_dir
        self.task_reports: List[TaskReport] = []

    async def run(self):
        for idx, task in enumerate(self.tasks, start=1):
            await self._run_one(task, idx)

        # Write summary.json
        summary_path = self.base_dir / "summary.json"
        summary_path.write_text(
            json.dumps([asdict(r) for r in self.task_reports], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Print console summary
        print("\n──────────  EXECUTION SUMMARY  ──────────")
        for r in self.task_reports:
            print(f"• {r.status.upper():9} | Task {r.number}: {r.task} → {r.result}")
        print(f"\nSaved JSON summary → {summary_path}\n")

    async def _run_one(self, task: Task, idx: int):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r'[<>:"/\\|?*]', "_", task.description)[:80].replace(" ", "_")
        task_dir = self.base_dir / f"task_{task.number:02d}_{safe}_{ts}"
        task_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        run_log = task_dir / "run_log.txt"
        planner_log = task_dir / "planner_log.txt"
        exec_log = task_dir / "executor_log.txt"
        lg = logging.getLogger(f"task_{task.number}")
        lg.setLevel(logging.INFO)
        fh = logging.FileHandler(run_log, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        lg.addHandler(fh)

        # Screenshot handler
        shot_dir = task_dir / "screenshots"
        js_log = task_dir / "log.json"
        sh = ScreenshotHandler(js_log, shot_dir)
        sh.setLevel(logging.INFO)
        for name in ["", "browser_use", "agent", "controller"]:
            logging.getLogger(name).addHandler(sh)

        lg.info(f"Task {task.number}: {task.description}")

        # Adjust task description based on dependency
        if task.prerequisite is None:
            effective_description = f"Navigate to Google Cloud Console “Home” page and then {task.description}"
        else:
            effective_description = task.description

        # Create agent with appropriate description
        plan_cb = LLMLogCallback(planner_log)
        exec_cb = LLMLogCallback(exec_log)
        exec_llm = ChatVertexAI(
            model_name="gemini-2.5-pro-preview-03-25",
            streaming=True,
            temperature=0.5,
            callbacks=[exec_cb],
            max_retries=6,
        )
        plan_llm = ChatVertexAI(
            model_name="gemini-2.5-pro-preview-03-25",
            streaming=True,
            temperature=1.0,
            callbacks=[plan_cb],
            max_retries=6,
        )
        agent = Agent(
            task=effective_description,
            llm=exec_llm,
            planner_llm=plan_llm,
            planner_interval=1,
            is_planner_reasoning=False,
            browser=self.browser,
            browser_context=self.browser_context,
            close_browser_on_run=False,
            enable_memory=False,
        )

        status = "completed"
        result_msg = "success"
        try:
            history = await agent.run()
            (task_dir / "run_result.json").write_text(
                json.dumps(serialize_history(history), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            status = "failed"
            result_msg = str(exc).split("\n")[0]
            tb = task_dir / "traceback.txt"
            tb.write_text("".join(traceback.format_exception(exc)), encoding="utf-8")
            lg.exception("Task encountered an error")

        # Record report
        self.task_reports.append(TaskReport(number=task.number, task=task.description, result=result_msg, status=status))

        # Clean up handlers
        for name in ["", "browser_use", "agent", "controller"]:
            logging.getLogger(name).removeHandler(sh)
        lg.removeHandler(fh)

# ---------------------------------------------------------------------------
# 🏁 Main driver
# ---------------------------------------------------------------------------

async def main():
    shared_ctx = None
    shared_browser = None
    try:
        # 1️⃣ Initial “login” agent
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = BASE_DIR / f"run_{run_ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        planner_log = run_dir / "planner_log.txt"
        executor_log = run_dir / "executor_log.txt"
        run_log = run_dir / "run_log.txt"

        run_logger = logging.getLogger("initial_run")
        run_logger.setLevel(logging.INFO)
        rh = logging.FileHandler(run_log, encoding="utf-8")
        rh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        run_logger.addHandler(rh)

        plan_cb = LLMLogCallback(planner_log)
        exec_cb = LLMLogCallback(executor_log)

        exec_llm = ChatVertexAI(
            model_name="gemini-2.5-pro-preview-03-25",
            temperature=0.5,
            callbacks=[exec_cb],
            max_retries=6,
        )
        plan_llm = ChatVertexAI(
            model_name="gemini-2.5-pro-preview-03-25",
            temperature=1.0,
            callbacks=[plan_cb],
            max_retries=6,
        )

        first_agent = Agent(
            task="Navigate to Google Cloud Platform and go to sign in",
            llm=exec_llm,
            planner_llm=plan_llm,
            planner_interval=1,
            use_vision_for_planner=False,
            is_planner_reasoning=False,
            close_browser_on_run=False,
            enable_memory=False,
        )

        shot_dir = run_dir / "screenshots"
        shot_handler = ScreenshotHandler(run_dir / "log.json", shot_dir)
        shot_handler.setLevel(logging.INFO)
        for name in ["", "browser_use", "agent", "controller"]:
            logging.getLogger(name).addHandler(shot_handler)

        run_logger.info("Task Started: Navigate to Google Cloud Platform and go to sign in")
        await first_agent.run()

        print("\n👉 Please complete the GCP login in the opened browser, then press Enter …")
        input()
        run_logger.info("User logged in to GCP")

        shared_browser = first_agent.browser
        shared_ctx = first_agent.browser_context

        # Remove login handlers
        for name in ["", "browser_use", "agent", "controller"]:
            logging.getLogger(name).removeHandler(shot_handler)
        run_logger.removeHandler(rh)
        del first_agent

        # 2️⃣ Load tasks and run with AdministratorAgent
        tasks_file = Path(__file__).parent / 'tasks.xlsx'
        tasks = load_tasks(tasks_file)
        admin = AdministratorAgent(
            tasks=tasks,
            browser=shared_browser,
            browser_context=shared_ctx,
            base_dir=BASE_DIR,
        )
        await admin.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Graceful shutdown with delay for background tasks
        await asyncio.sleep(5)  # Wait 5 seconds for tasks like screenshot logging to complete
        if shared_ctx is not None:
            await shared_ctx.close()
        if shared_browser is not None:
            await shared_browser.close()

if __name__ == "__main__":
    asyncio.run(main())