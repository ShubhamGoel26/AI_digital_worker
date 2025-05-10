#!/usr/bin/env python
"""
Interactive *Planner ⇄ Executor* demo for Google Cloud Platform automation.
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
This file shows **exactly where** the *high‑level planner* (`gpt‑4o`) and the
*executing agent* (`gpt‑4.1‑mini`) exchange JSON messages so you can follow the
feedback loop step‑by‑step.

ASCII flow‑chart of the queues
------------------------------

      ┌─────────────┐        JSON task         ┌──────────────┐
      │             │   (next browser action)  │              │
      │  Planner    │  ═══════════════════════▶│  Executor    │
      │  coroutine  │                          │  coroutine   │
      │             │◀═════════════════════════│              │
      └─────────────┘        JSON feedback     └──────────────┘

Queues used
~~~~~~~~~~~
• **q_planner_to_exec** : task → executor  (one dict per turn)
• **q_exec_to_planner** : feedback → planner (string/JSON)

Outer components
~~~~~~~~~~~~~~~~
1. `human_loop()` injects *new high‑level intents* (what **you** type) into the
   feedback queue so the planner hears them.
2. A *single Playwright browser* lives for the entire session (log in once).
3. `ScreenshotHandler` still captures each key step.

— Shubham Goel • April 2025
"""

import asyncio
import json
import logging
import os
import sys
from asyncio import Queue
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mss import mss
from PIL import Image

from browser_use import Agent

# ================================================================
# 1.  ENV & directories
# ================================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

SCREENSHOT_DIR = 'screenshots'
DEBUG_LOG_DIR = 'debug_logs'
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)

# ================================================================
# 2.  LLMs
# ================================================================
exec_llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0.5)  # executor brain
plan_llm = ChatOpenAI(model='gpt-4o', temperature=1.0)  # planner brain


# ================================================================
# 3.  Screenshot handler (unchanged apart from minor tidy‑ups)
# ================================================================
class ScreenshotHandler(logging.Handler):
	"""Grab 1200×800 PNGs whenever browser_use emits an interesting log."""

	def __init__(self, log_file: str, screenshot_dir: str):
		super().__init__()
		self.log_file, self.screenshot_dir = log_file, screenshot_dir
		self.step = 1

	async def _capture(self, filename: str):
		await asyncio.sleep(3)  # allow DOM paint
		with mss() as sct:
			mon = {'top': 100, 'left': 100, 'width': 1200, 'height': 800}
			shot = sct.grab(mon)
			Image.frombytes('RGB', (shot.width, shot.height), shot.rgb).save(filename)

	def emit(self, record):
		if not any(t in record.getMessage() for t in ('Action 1/1', 'Next goal', 'Clicked', 'User logged in')):
			return
		ts = datetime.now().strftime('%Y%m%d_%H%M%S')
		fn = f'{self.screenshot_dir}/step_{self.step}_{ts}.png'
		asyncio.create_task(self._capture(fn))
		json.dump({'step': self.step, 'msg': record.getMessage(), 'shot': fn}, open(self.log_file, 'a', encoding='utf-8'))
		print(f'Step {self.step}: screenshot saved → {fn}')
		self.step += 1


# ================================================================
# 4‑A. High‑level PLANNER coroutine  (plan  → q_planner_to_exec)
# ================================================================
async def planner(initial_goal: str, q_planner_to_exec: Queue, q_exec_to_planner: Queue):
	"""Continuously decide the *next* browser action and push it to the executor.

	Feedback cycle:
	1. Wait for previous feedback from **executor** (first cycle uses "").
	2. Call `plan_llm` to choose one JSON task.
	3. `await q_planner_to_exec.put(task)`  ➜ **executor**.
	4. If `terminate`, exit; otherwise wait for fresh feedback and loop.
	"""
	feedback = ''
	while True:
		# ── 2️⃣ LLM thinks  ──────────────────────────────────────────
		resp = await plan_llm.ainvoke(
			[
				{
					'role': 'system',
					'content': (
						'You are the high‑level planner. Take the last browser feedback,'
						' decide ONE next action, return JSON {task, terminate}.'
					),
				},
				{'role': 'user', 'content': f'feedback: {feedback}'},
			]
		)
		try:
			plan = json.loads(resp.content.strip())
		except json.JSONDecodeError:
			plan = {'task': resp.content.strip(), 'terminate': False}

		# ── 3️⃣ send task → executor  ────────────────────────────────
		await q_planner_to_exec.put(plan)  # ★ main outbound path ★

		if plan.get('terminate'):
			break  # graceful shutdown

		# ── 1️⃣ wait for executor feedback  ─────────────────────────
		feedback = await q_exec_to_planner.get()  # ★ main inbound path ★


# ================================================================
# 4‑B. EXECUTOR coroutine  (takes tasks, runs browser_use.Agent)
# ================================================================
async def executor(q_planner_to_exec: Queue, q_exec_to_planner: Queue, browser, browser_ctx, shandler: ScreenshotHandler):
	"""Run each task in Playwright and push a compact summary back to planner."""
	while True:
		# ── receive next task from planner ─────────────────────────
		plan = await q_planner_to_exec.get()
		if plan.get('terminate'):
			break
		task_desc = plan.get('task', '')

		agent = Agent(
			task=task_desc,
			llm=exec_llm,
			# Embedded micro‑planner to break clicks/inputs down further ↓
			planner_llm=plan_llm,
			planner_interval=1,
			is_planner_reasoning=False,
			browser=browser,
			browser_context=browser_ctx,
			close_browser_on_run=False,
			enable_memory=False,
		)
		try:
			await agent.run()
			summary = f'✅ {task_desc}'
		except Exception as e:
			summary = f'❌ {task_desc} → {e}'
		finally:
			del agent

		# ── send feedback back to planner ──────────────────────────
		await q_exec_to_planner.put(summary)  # ★ feedback funnel ★


# ================================================================
# 5.  Main entry‑point
# ================================================================
async def main():
	# ── screenshots ------------------------------------------------
	sh = ScreenshotHandler(f'{SCREENSHOT_DIR}/log.json', SCREENSHOT_DIR)
	sh.setLevel(logging.INFO)
	for n in ('', 'browser_use', 'agent', 'controller'):
		logging.getLogger(n).addHandler(sh)

	# ── one‑time login --------------------------------------------
	boot = Agent(
		task='Open https://cloud.google.com and wait at sign‑in',
		llm=exec_llm,
		planner_llm=plan_llm,
		planner_interval=1,
		close_browser_on_run=False,
		enable_memory=False,
	)
	await boot.run()
	input('\nManually log in, then press <Enter> …')
	logging.info('User logged in to GCP')

	# shared Playwright instances
	browser, ctx = boot.browser, boot.browser_context
	del boot

	# ── create the two queues -------------------------------------
	q_planner_to_exec: Queue = Queue()  # tasks  → executor
	q_exec_to_planner: Queue = Queue()  # feedback → planner

	# spawn coroutines
	tasks = [
		planner('Handle each human request until they type exit', q_planner_to_exec, q_exec_to_planner),
		executor(q_planner_to_exec, q_exec_to_planner, browser, ctx, sh),
	]

	async def human_loop():
		while True:
			cmd = input("\nNext GCP action (or 'exit'): ").strip()
			if cmd.lower() == 'exit':
				await q_planner_to_exec.put({'task': 'Session done', 'terminate': True})
				break
			# Push the new desire into *feedback* so planner hears it next turn.
			await q_exec_to_planner.put(f'💬 human wants: {cmd}')

	try:
		await asyncio.gather(*tasks, human_loop())
	finally:
		for n in ('', 'browser_use', 'agent', 'controller'):
			logging.getLogger(n).removeHandler(sh)
		await ctx.close()
		await browser.close()


# ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
	asyncio.run(main())
