#!/usr/bin/env python
# gcp_manual_login_planner_loop.py
# ---------------------------------------------------------------
#  Planner  (reasoning) : gpt‑4o
#  Executor (clicks)    : gpt‑4.1‑mini
# ---------------------------------------------------------------

import os, sys, asyncio, json, logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from browser_use import Agent
from mss import mss
from PIL import Image

# -----------------------------------------------------------------
# 1.  ENV & directories
# -----------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

SS_DIR, DBG_DIR = "screenshots", "debug_logs"
os.makedirs(SS_DIR,  exist_ok=True)
os.makedirs(DBG_DIR, exist_ok=True)

# -----------------------------------------------------------------
# 2.  LLMs
# -----------------------------------------------------------------
EXEC_LLM = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)
PLAN_LLM = ChatOpenAI(model="gpt-4o",  temperature=1.2)

# -----------------------------------------------------------------
# 3.  Screenshot handler
# -----------------------------------------------------------------
class ScreenshotHandler(logging.Handler):
    def __init__(self, log_file, ss_dir):
        super().__init__()
        self.log_file, self.ss_dir = log_file, ss_dir
        self.step, self.last_goal  = 1, ""
        self.debug_file = os.path.join(DBG_DIR, "debug_log.txt")

    async def _cap(self, fn):
        await asyncio.sleep(3)
        with mss() as sct:
            mon = {"top": 100, "left": 100, "width": 1200, "height": 800}
            img = Image.frombytes("RGB", (mon["width"], mon["height"]),
                                  sct.grab(mon).rgb)
            img.save(fn, "PNG")

    def emit(self, record):
        msg = record.getMessage()
        with open(self.debug_file, "a", encoding="utf-8") as f:
            f.write(f"DEBUG: {msg}\n")
        if any(k in msg for k in ["Action 1/1", "Next goal", "Clicked",
                                  "User logged in"]):
            fn = f"{self.ss_dir}/step_{self.step}_{datetime.now():%Y%m%d_%H%M%S}.png"
            asyncio.create_task(self._cap(fn)); self.step += 1

# -----------------------------------------------------------------
# 4.  Planner helper
# -----------------------------------------------------------------
SYSTEM_PLAN = """
You are the high‑level planner for a Google‑Cloud browser agent.
You receive:
  • USER_GOAL           – the ultimate task (e.g. "create bucket 'demo‑1'")
  • PAGE_SUMMARY        – clickable elements text of current page
  • EXECUTOR_FEEDBACK   – success/error from last low‑level action (optional)

Return valid JSON:
{
 "next_step": "Single UI instruction for executor",
 "done": false/true,
 "reasoning": "brief"
}
Keep steps granular (one click/type/scroll at a time). Set done=true
only when the user goal is fully satisfied.
"""

async def ask_planner(goal, page_summary, feedback=None):
    msgs = [
        SystemMessage(content=SYSTEM_PLAN),
        HumanMessage(content=f"USER_GOAL:\n{goal}"),
        HumanMessage(content=f"PAGE_SUMMARY:\n{page_summary[:4000]}"),
    ]
    if feedback:
        msgs.append(HumanMessage(content=f"EXECUTOR_FEEDBACK:\n{feedback}"))
    resp = await PLAN_LLM.ainvoke(msgs)
    try:
        return json.loads(str(resp.content))
    except Exception:
        raise RuntimeError(f"Planner gave invalid JSON:\n{resp.content}")

# -----------------------------------------------------------------
# 5.  Main coroutine
# -----------------------------------------------------------------
async def main():
    # logging
    handler = ScreenshotHandler("log.json", SS_DIR)
    handler.setLevel(logging.INFO)
    for n in ["", "browser_use", "agent", "controller"]:
        logging.getLogger(n).addHandler(handler)

    # ---- Login phase ------------------------------------------------------
    login_agent = Agent(
        task="Navigate to Google Cloud Platform and go to sign in",
        llm=EXEC_LLM, close_browser_on_run=False, enable_memory=False
    )
    await login_agent.run()
    input("\n🛂  Log in to GCP in the browser, then press Enter…")
    logging.info("User logged in")

    browser, ctx = login_agent.browser, login_agent.browser_context
    del login_agent

    # ---- Interactive loop -------------------------------------------------
    try:
        while True:
            user_goal = input("\nNext GCP task (or 'exit'): ").strip()
            if user_goal.lower() == "exit":
                break
            if not user_goal:
                continue

            print(f"\n📋 Working on goal: {user_goal}")
            # persistent small executor
            exec_agent = Agent(
                task="stand‑by", llm=EXEC_LLM,
                browser=browser, browser_context=ctx,
                close_browser_on_run=False, enable_memory=False
            )

            feedback = None
            for round_no in range(20):
                state = await ctx.get_state()
                summary = state.element_tree.clickable_elements_to_string()

                plan = await ask_planner(user_goal, summary, feedback)
                if plan.get("done"):
                    print("✅  Planner says goal complete!")
                    break

                step = plan["next_step"]
                print(f"→  [{round_no+1}] executor will: {step}")
                exec_agent.task = step
                await exec_agent.run(max_steps=10)

                last = exec_agent.state.last_result[-1]
                feedback = last.error or last.extracted_content or "success"

            del exec_agent
    finally:
        for n in ["", "browser_use", "agent", "controller"]:
            logging.getLogger(n).removeHandler(handler)
        await ctx.close(); await browser.close()

# -----------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
