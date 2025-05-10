#!/usr/bin/env python
# gcp_manual_login_planner.py
# ---------------------------------------------------------------
#  ▸ execution  LLM : gpt‑4.1‑mini      (fast / cheap)
#  ▸ planning   LLM : gpt‑4o            (bigger / better reasoning)
# ---------------------------------------------------------------

import asyncio
import json
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from mss import mss
from PIL import Image

from browser_use import Agent

# -----------------------------------------------------------------
# 1.  ENV & directories
# -----------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

BASE_SCREENSHOT_DIR = 'screenshots'
DEBUG_LOG_DIR = 'debug_logs'
LLM_LOG_DIR = 'llm_logs'
os.makedirs(BASE_SCREENSHOT_DIR, exist_ok=True)
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
os.makedirs(LLM_LOG_DIR, exist_ok=True)

# Enable LangChain debug logging (local only)
os.environ['LANGCHAIN_DEBUG'] = '1'

# Set up logging for LangChain to a separate file
langchain_logger = logging.getLogger('langchain')
langchain_logger.setLevel(logging.DEBUG)
llm_log_file = os.path.join(LLM_LOG_DIR, f'langchain_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler = logging.FileHandler(llm_log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
langchain_logger.addHandler(file_handler)

# Add console handler for real-time debugging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
langchain_logger.addHandler(console_handler)


# -----------------------------------------------------------------
# 2.  Custom Callback Handler for LLM Prompts and Responses
# -----------------------------------------------------------------
class LLMLogCallback(BaseCallbackHandler):
	def __init__(self, log_file):
		self.log_file = log_file

	def on_llm_start(self, serialized, prompts, **kwargs):
		model_name = serialized.get('kwargs', {}).get('model_name', 'unknown')
		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write(f'\n=== LLM Prompt (Model: {model_name}) ===\n')
			for prompt in prompts:
				f.write(f'{prompt}\n')
			f.flush()

	def on_llm_end(self, response, **kwargs):
		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write('\n=== LLM Response ===\n')
			for generation in response.generations:
				for gen in generation:
					f.write(f'{gen.text}\n')
			f.write('\n')
			f.flush()

	def on_llm_error(self, error, **kwargs):
		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write(f'\n=== LLM Error ===\n{str(error)}\n\n')
			f.flush()


# -----------------------------------------------------------------
# 3.  LLMs with Callback
# -----------------------------------------------------------------
llm_callback = LLMLogCallback(llm_log_file)
exec_llm = ChatOpenAI(model='gpt-4.1', temperature=0.5, callbacks=[llm_callback])
plan_llm = ChatOpenAI(model='o4-mini', temperature=1.0, callbacks=[llm_callback])  # <─ bigger brain


# -----------------------------------------------------------------
# 4.  Screenshot helper (unchanged)
# -----------------------------------------------------------------
class ScreenshotHandler(logging.Handler):
	def __init__(self, log_file, screenshot_dir):
		super().__init__()
		self.log_file = log_file
		self.screenshot_dir = screenshot_dir
		self.step = 1
		self.last_goal = ''
		self.debug_file = os.path.join(DEBUG_LOG_DIR, 'debug_log.txt')

	async def _capture(self, filename):
		await asyncio.sleep(3)  # wait for page paint
		try:
			with mss() as sct:
				monitor = {'top': 100, 'left': 100, 'width': 1200, 'height': 800}
				shot = sct.grab(monitor)
				img = Image.frombytes('RGB', (shot.width, shot.height), shot.rgb)
				img.save(filename, 'PNG')
		except Exception as e:
			print(f'Error capturing screenshot: {e}')

	def emit(self, record):
		msg = record.getMessage()
		with open(self.debug_file, 'a', encoding='utf-8') as f:
			f.write(f"DEBUG: Checking message: '{msg}'\n")

		triggers = [
			'Action 1/1:',
			'Next goal:',
			'Searched for',
			'Navigating to',
			'Entering',
			'Clicking',
			'User logged in',
			'Clicked',
		]
		if any(t in msg for t in triggers):
			ts = datetime.now().strftime('%Y%m%d_%H%M%S')
			file = f'{self.screenshot_dir}/step_{self.step}_{ts}.png'
			asyncio.create_task(self._capture(file))

			entry = {
				'step': self.step,
				'action': msg
				if any(
					t in msg
					for t in ['Action', 'Navigating', 'Entering', 'Clicking', 'Searched for', 'User logged in', 'Clicked']
				)
				else '',
				'next_goal': msg if 'Next goal' in msg else self.last_goal,
				'screenshot': file,
			}
			with open(self.log_file, 'a', encoding='utf-8') as f:
				json.dump(entry, f)
				f.write('\n')

			if 'Next goal:' in msg:
				self.last_goal = msg
			print(f'Step {self.step}: Saved {file}')
			self.step += 1


# -----------------------------------------------------------------
# 5.  Main coroutine
# -----------------------------------------------------------------
async def main():
	# ----- logging / screenshots for first agent ------------------
	ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	first_task_dir = os.path.join(BASE_SCREENSHOT_DIR, f'initial_login_{ts}')
	os.makedirs(first_task_dir, exist_ok=True)
	log_json = os.path.join(first_task_dir, 'log.json')
	handler = ScreenshotHandler(log_json, first_task_dir)
	handler.setLevel(logging.INFO)
	for name in ['', 'browser_use', 'agent', 'controller']:
		logging.getLogger(name).addHandler(handler)

	# -------------------------------------------------------------
	# 5‑A  First agent: open GCP sign‑in page
	# -------------------------------------------------------------
	first_agent = Agent(
		task='Navigate to Google Cloud Platform and go to sign in',
		llm=exec_llm,
		planner_llm=plan_llm,
		planner_interval=1,
		use_vision_for_planner=False,
		is_planner_reasoning=False,
		close_browser_on_run=False,
		enable_memory=False,
	)
	# Log task start
	with open(llm_log_file, 'a', encoding='utf-8') as f:
		f.write('\n=== Task Started: Navigate to Google Cloud Platform and go to sign in ===\n')
		f.flush()

	# Run the agent
	with open(llm_log_file, 'a', encoding='utf-8') as f:
		f.write('\n=== Running Agent for Task ===\n')
		f.flush()
	history = await first_agent.run()

	# Log agent history
	with open(llm_log_file, 'a', encoding='utf-8') as f:
		f.write('\n=== Agent History After Run ===\n')
		f.write(json.dumps(history, indent=2))
		f.write('\n')
		f.flush()

	# Log browser state after run
	try:
		# Access the browser context and get the first page
		context = first_agent.browser_context
		pages = await context.pages()  # Use async method to get pages
		if not pages:
			page = await context.new_page()
			await page.goto('about:blank')  # Ensure a page is open
		else:
			page = pages[0]
		current_url = await page.evaluate('() => window.location.href')
		with open(llm_log_file, 'a', encoding='utf-8') as f:
			f.write(f'\n=== Browser State After Run: URL = {current_url} ===\n')
			f.flush()
	except Exception as e:
		with open(llm_log_file, 'a', encoding='utf-8') as f:
			f.write(f'\n=== Browser State Error: {str(e)} ===\n')
			f.flush()

	print('\nLog in manually, then press Enter …')
	input()
	logging.info('User logged in to GCP')

	shared_browser = first_agent.browser
	shared_browser_ctx = first_agent.browser_context
	del first_agent

	# Remove handler for first agent
	for name in ['', 'browser_use', 'agent', 'controller']:
		logging.getLogger(name).removeHandler(handler)

	# -------------------------------------------------------------
	# 5‑B  Command loop
	# -------------------------------------------------------------
	try:
		while True:
			act = input("\nNext GCP action (or 'exit'): ").strip()
			if act.lower() == 'exit':
				break

			# Create new screenshot folder and log file for each task
			ts = datetime.now().strftime('%Y%m%d_%H%M%S')
			safe_task_name = ''.join(c if c.isalnum() else '_' for c in act[:30])
			task_dir = os.path.join(BASE_SCREENSHOT_DIR, f'task_{safe_task_name}_{ts}')
			os.makedirs(task_dir, exist_ok=True)
			task_log_json = os.path.join(task_dir, 'log.json')

			# Set up new screenshot handler for this task
			task_handler = ScreenshotHandler(task_log_json, task_dir)
			task_handler.setLevel(logging.INFO)
			for name in ['', 'browser_use', 'agent', 'controller']:
				logging.getLogger(name).addHandler(task_handler)

			fresh_agent = Agent(
				task=act,
				llm=exec_llm,
				planner_llm=plan_llm,
				planner_interval=1,
				is_planner_reasoning=False,
				browser=shared_browser,
				browser_context=shared_browser_ctx,
				close_browser_on_run=False,
				enable_memory=False,
			)
			# Log task start
			with open(llm_log_file, 'a', encoding='utf-8') as f:
				f.write(f'\n=== Task Started: {act} ===\n')
				f.flush()

			# Run the agent
			with open(llm_log_file, 'a', encoding='utf-8') as f:
				f.write('\n=== Running Agent for Task ===\n')
				f.flush()
			history = await fresh_agent.run()

			# Log agent history
			with open(llm_log_file, 'a', encoding='utf-8') as f:
				f.write('\n=== Agent History After Run ===\n')
				f.write(json.dumps(history, indent=2))
				f.write('\n')
				f.flush()

			# Log browser state after run
			try:
				context = fresh_agent.browser_context
				pages = await context.pages()
				if not pages:
					page = await context.new_page()
					await page.goto('about:blank')
				else:
					page = pages[0]
				current_url = await page.evaluate('() => window.location.href')
				with open(llm_log_file, 'a', encoding='utf-8') as f:
					f.write(f'\n=== Browser State After Run: URL = {current_url} ===\n')
					f.flush()
			except Exception as e:
				with open(llm_log_file, 'a', encoding='utf-8') as f:
					f.write(f'\n=== Browser State Error: {str(e)} ===\n')
					f.flush()

			# Clean up: remove handler after task
			for name in ['', 'browser_use', 'agent', 'controller']:
				logging.getLogger(name).removeHandler(task_handler)
			del fresh_agent

	finally:
		await shared_browser_ctx.close()
		await shared_browser.close()
		# Close the LangChain log file handler
		langchain_logger.removeHandler(file_handler)
		langchain_logger.removeHandler(console_handler)
		file_handler.close()


# -----------------------------------------------------------------
# 6.  Entry‑point
# -----------------------------------------------------------------
if __name__ == '__main__':
	asyncio.run(main())
