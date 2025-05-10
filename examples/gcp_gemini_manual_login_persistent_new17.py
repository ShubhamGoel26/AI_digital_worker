#!/usr/bin/env python
# gcp_gemini_manual_login_persistent_new16.py
# ---------------------------------------------------------------
#  ▸ execution  LLM : gemini-2.5-pro-preview-03-25 (via Vertex AI)
#  ▸ planning   LLM : o4-mini              (bigger / better reasoning)
#  ▸ Gemini integration via Vertex AI
# ---------------------------------------------------------------

import asyncio
import json
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from google.cloud import aiplatform as vertexai  # Import Vertex AI SDK
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI  # Import for Vertex AI models
from mss import mss
from PIL import Image

from browser_use import Agent

# -----------------------------------------------------------------
# 1.  ENV & directories
# -----------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

BASE_DIR = 'runs'
os.makedirs(BASE_DIR, exist_ok=True)

# Enable LangChain debug logging (local only)
os.environ['LANGCHAIN_DEBUG'] = '1'

# Get Google Cloud project ID from environment variable
project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
if not project_id:
	raise ValueError(
		'GOOGLE_CLOUD_PROJECT environment variable not set. '
		"Please set it using 'set GOOGLE_CLOUD_PROJECT=your-project-id' (Windows) "
		"or 'export GOOGLE_CLOUD_PROJECT=your-project-id' (Unix), "
		"or pass it directly in vertexai.init(project='your-project-id')."
	)

# Initialize Vertex AI (required for Gemini model)
try:
	vertexai.init(project=project_id, location='us-central1')
except Exception as e:
	raise Exception(
		f'Failed to initialize Vertex AI. Ensure credentials are set up correctly. '
		f'See https://cloud.google.com/docs/authentication/external/set-up-adc for more information. '
		f'Error: {str(e)}'
	)


# -----------------------------------------------------------------
# 2.  Helper to serialize AgentHistoryList
# -----------------------------------------------------------------
def serialize_history(history):
	"""Convert AgentHistoryList to a serializable dictionary."""
	try:
		serializable_history = []
		for item in history:
			if hasattr(item, '__dict__'):
				serializable_history.append(item.__dict__)
			elif isinstance(item, (dict, list, str, int, float, bool, type(None))):
				serializable_history.append(item)
			else:
				serializable_history.append(str(item))
		return serializable_history
	except Exception as e:
		return {'error': f'Failed to serialize history: {str(e)}'}


# -----------------------------------------------------------------
# 3.  Custom Callback Handler for LLM Prompts and Responses
# -----------------------------------------------------------------
class LLMLogCallback(BaseCallbackHandler):
	def __init__(self, log_file, global_log_file, agent_id):
		self.log_file = log_file
		self.global_log_file = global_log_file
		self.agent_id = agent_id

	def on_llm_start(self, serialized, prompts, **kwargs):
		model_name = serialized.get('kwargs', {}).get('model_name', 'unknown')
		# Log to agent-specific file
		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write(f'\n=== LLM Prompt (Model: {model_name}) ===\n')
			for prompt in prompts:
				f.write(f'{prompt}\n')
			f.flush()
		# Log to global file with agent_id
		with open(self.global_log_file, 'a', encoding='utf-8') as f:
			f.write(f'\n=== {self.agent_id} - LLM Prompt (Model: {model_name}) ===\n')
			for prompt in prompts:
				f.write(f'{prompt}\n')
			f.flush()

	def on_llm_end(self, response, **kwargs):
		# Log to agent-specific file
		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write('\n=== LLM Response ===\n')
			try:
				f.write(f'Response type: {type(response)}\n')
				if hasattr(response, 'generations'):
					if not response.generations:
						f.write('No generations found in response\n')
					else:
						for generation_list in response.generations:
							for gen in generation_list:
								if hasattr(gen, 'message') and hasattr(gen.message, 'content'):
									content = gen.message.content
									f.write(f'Content length: {len(content)}\n')
									if content:
										try:
											json_response = json.loads(content)
											formatted_json = json.dumps(json_response, indent=4, ensure_ascii=False)
											f.write(f'{formatted_json}\n')
										except json.JSONDecodeError:
											f.write(f'{content}\n')
									else:
										f.write('Empty content\n')
								else:
									f.write("Generation lacks 'message' or 'content' attribute\n")
				else:
					if hasattr(response, 'content'):
						content = response.content
						f.write(f'Content length: {len(content)}\n')
						if content:
							try:
								json_response = json.loads(content)
								formatted_json = json.dumps(json_response, indent=4, ensure_ascii=False)
								f.write(f'{formatted_json}\n')
							except json.JSONDecodeError:
								f.write(f'{content}\n')
						else:
							f.write('Empty content\n')
					else:
						f.write("Response lacks 'generations' or 'content' attributes\n")
			except Exception as e:
				f.write(f'Error logging response: {str(e)}\n')
			f.write('\n')
			f.flush()

		# Log to global file with agent_id
		with open(self.global_log_file, 'a', encoding='utf-8') as f:
			f.write(f'\n=== {self.agent_id} - LLM Response ===\n')
			try:
				f.write(f'Response type: {type(response)}\n')
				if hasattr(response, 'generations'):
					if not response.generations:
						f.write('No generations found in response\n')
					else:
						for generation_list in response.generations:
							for gen in generation_list:
								if hasattr(gen, 'message') and hasattr(gen.message, 'content'):
									content = gen.message.content
									f.write(f'Content length: {len(content)}\n')
									if content:
										try:
											json_response = json.loads(content)
											formatted_json = json.dumps(json_response, indent=4, ensure_ascii=False)
											f.write(f'{formatted_json}\n')
										except json.JSONDecodeError:
											f.write(f'{content}\n')
									else:
										f.write('Empty content\n')
								else:
									f.write("Generation lacks 'message' or 'content' attribute\n")
				else:
					if hasattr(response, 'content'):
						content = response.content
						f.write(f'Content length: {len(content)}\n')
						if content:
							try:
								json_response = json.loads(content)
								formatted_json = json.dumps(json_response, indent=4, ensure_ascii=False)
								f.write(f'{formatted_json}\n')
							except json.JSONDecodeError:
								f.write(f'{content}\n')
						else:
							f.write('Empty content\n')
					else:
						f.write("Response lacks 'generations' or 'content' attributes\n")
			except Exception as e:
				f.write(f'Error logging response: {str(e)}\n')
			f.write('\n')
			f.flush()

	def on_llm_error(self, error, **kwargs):
		# Log to agent-specific file
		with open(self.log_file, 'a', encoding='utf-8') as f:
			f.write(f'\n=== LLM Error ===\n{str(error)}\n\n')
			f.flush()
		# Log to global file with agent_id
		with open(self.global_log_file, 'a', encoding='utf-8') as f:
			f.write(f'\n=== {self.agent_id} - LLM Error ===\n{str(error)}\n\n')
			f.flush()


# -----------------------------------------------------------------
# 4.  Screenshot helper
# -----------------------------------------------------------------
class ScreenshotHandler(logging.Handler):
	def __init__(self, log_file, screenshot_dir):
		super().__init__()
		self.log_file = log_file
		self.screenshot_dir = screenshot_dir
		self.step = 1
		self.last_goal = ''
		self.debug_file = os.path.join(self.screenshot_dir, 'debug_log.txt')

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
	# Create unique run folder
	run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	run_dir = os.path.join(BASE_DIR, f'run_{run_ts}')
	os.makedirs(run_dir, exist_ok=True)

	# Set up log files for planner and executor
	planner_log_file = os.path.join(run_dir, 'planner_log.txt')
	executor_log_file = os.path.join(run_dir, 'executor_log.txt')
	run_log_file = os.path.join(run_dir, 'run_log.txt')

	# Create global LLM log file
	global_llm_log_file = os.path.join(run_dir, 'all_llm_calls.log')

	# Set up general run logger
	run_logger = logging.getLogger('run')
	run_logger.setLevel(logging.INFO)
	run_file_handler = logging.FileHandler(run_log_file, encoding='utf-8')
	run_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
	run_logger.addHandler(run_file_handler)

	# Set up LLM callbacks with global_log_file and agent_id
	planner_callback = LLMLogCallback(planner_log_file, global_llm_log_file, 'main_agent')
	executor_callback = LLMLogCallback(executor_log_file, global_llm_log_file, 'main_agent')

	# -----------------------------------------------------------------
	# 5‑A  First agent: open GCP sign‑in page
	# -----------------------------------------------------------------
	# Use ChatVertexAI for executor LLM (Gemini model)
	try:
		exec_llm = ChatVertexAI(model='gemini-2.5-pro-preview-03-25', temperature=0.5, callbacks=[executor_callback])
	except Exception as e:
		raise Exception(
			f"Failed to initialize executor LLM with model 'gemini-2.5-pro-preview-03-25'. "
			f'This model may not be available or accessible in your project. '
			f'Please verify the model name and ensure your project has access to it. '
			f'Check the Vertex AI Model Registry in the Google Cloud Console, or contact Google Cloud support to request access. '
			f'Error: {str(e)}'
		)

	# Planner LLM remains as specified
	try:
		plan_llm = ChatVertexAI(model='gemini-2.5-pro-preview-03-25', temperature=1.0, callbacks=[planner_callback])
	except Exception as e:
		raise Exception(
			f"Failed to initialize planner LLM with model 'o4-mini'. "
			f'This model may not be available or accessible in your project. '
			f'Please verify the model name and ensure your project has access to it. '
			f'Check the Vertex AI Model Registry in the Google Cloud Console, or contact Google Cloud support to request access. '
			f'Error: {str(e)}'
		)

	first_agent = Agent(
		task='Navigate to Google Cloud Platform and go to sign in',
		llm=exec_llm,
		planner_llm=plan_llm,
		planner_interval=1,
		use_vision_for_planner=False,
		close_browser_on_run=False,
		enable_memory=False,
	)

	# Set up screenshot handler for this run
	screenshot_dir = os.path.join(run_dir, 'screenshots')
	os.makedirs(screenshot_dir, exist_ok=True)
	log_json = os.path.join(run_dir, 'log.json')
	handler = ScreenshotHandler(log_json, screenshot_dir)
	handler.setLevel(logging.INFO)
	for name in ['', 'browser_use', 'agent', 'controller']:
		logging.getLogger(name).addHandler(handler)

	# Log task start
	run_logger.info('Task Started: Navigate to Google Cloud Platform and go to sign in')

	# Run the agent
	run_logger.info('Running Agent for Task')
	history = await first_agent.run()

	# Executor LLM Action Logging
	with open(executor_log_file, 'a', encoding='utf-8') as f:
		f.write('\n=== Executor Actions from History (Initial Run) ===\n')
		serialized_history = serialize_history(history)
		for i, entry in enumerate(serialized_history):
			f.write(f'\nStep {i + 1}:\n')
			if isinstance(entry, dict):
				if 'action' in entry:
					f.write(f'  Action: {json.dumps(entry["action"], indent=4, ensure_ascii=False)}\n')
				if 'llm_output' in entry:
					f.write(f'  LLM Output: {json.dumps(entry["llm_output"], indent=4, ensure_ascii=False)}\n')
				for key, value in entry.items():
					if key not in ['action', 'llm_output']:
						f.write(f'  {key}: {value}\n')
			else:
				f.write(f'  {entry}\n')
		f.write('\n')

	# Serialize and log agent history
	with open(os.path.join(run_dir, 'run_result.json'), 'w', encoding='utf-8') as f:
		json.dump(serialized_history, f, indent=2)

	# Log browser state
	try:
		# Access the browser from the context and get the current page
		browser = first_agent.browser
		if browser is not None:
			# Create a new context if none exists
			context = first_agent.browser_context
			if context is None:
				context = await browser.new_context()
			# Get the current page or create a new one
			page = await context.new_page() if not hasattr(context, 'pages') or not context.pages else context.pages[0]
			if page.url == 'about:blank':
				await page.goto('https://cloud.google.com')
			current_url = page.url
		else:
			current_url = 'unknown'
		run_logger.info(f'Browser State After Run: URL = {current_url}')
	except Exception as e:
		run_logger.error(f'Browser State Error: {str(e)}')
		current_url = 'unknown'

	# Force executor LLM call
	try:
		prompt = f"""
        Current URL: {current_url}
        Task: Verify that the Google Cloud Platform sign-in page is loaded and identify the next action.
        Instructions: Confirm if the sign-in interface is present. If so, suggest the next action. Return JSON:
        {{
            "current_state": {{"evaluation": "Description of page state"}},
            "next_action": "Suggested action"
        }}
        """
		response = await exec_llm.ainvoke([HumanMessage(content=prompt)])
		with open(executor_log_file, 'a', encoding='utf-8') as f:
			f.write('\n=== Forced Executor LLM Call ===\n')
			f.write(f'Prompt: {prompt}\n')
			if hasattr(response, 'content'):
				f.write(f'Response: {response.content}\n')
			else:
				f.write('No response content\n')
			f.flush()
	except Exception as e:
		with open(executor_log_file, 'a', encoding='utf-8') as f:
			f.write(f'\n=== Forced Executor LLM Error: {str(e)} ===\n')
			f.flush()

	print('\nLog in manually, then press Enter …')
	input()
	run_logger.info('User logged in to GCP')

	shared_browser = first_agent.browser
	shared_browser_ctx = first_agent.browser_context
	del first_agent

	# Remove handler
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

			# Create task run folder
			run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
			task_run_dir = os.path.join(BASE_DIR, f'task_{act.replace(" ", "_")}_{run_ts}')
			os.makedirs(task_run_dir, exist_ok=True)

			# Set up log files
			task_planner_log = os.path.join(task_run_dir, 'planner_log.txt')
			task_executor_log = os.path.join(task_run_dir, 'executor_log.txt')
			task_run_log = os.path.join(task_run_dir, 'run_log.txt')

			# Set up task logger
			task_run_logger = logging.getLogger(f'task_run_{act}')
			task_run_logger.setLevel(logging.INFO)
			task_file_handler = logging.FileHandler(task_run_log, encoding='utf-8')
			task_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
			task_run_logger.addHandler(task_file_handler)

			# Set up LLM callbacks
			task_planner_callback = LLMLogCallback(task_planner_log, global_llm_log_file, f'task_agent_{act}')
			task_executor_callback = LLMLogCallback(task_executor_log, global_llm_log_file, f'task_agent_{act}')

			# Create new agent
			try:
				exec_llm_task = ChatVertexAI(
					model='gemini-2.5-pro-preview-03-25', temperature=0.5, callbacks=[task_executor_callback]
				)
			except Exception as e:
				raise Exception(
					f"Failed to initialize executor LLM with model 'gemini-2.5-pro-preview-03-25' for task agent. "
					f'This model may not be available or accessible in your project. '
					f'Check the Vertex AI Model Registry in the Google Cloud Console, or contact Google Cloud support to request access. '
					f'Error: {str(e)}'
				)

			try:
				plan_llm_task = ChatVertexAI(model='o4-mini', temperature=1.0, callbacks=[task_planner_callback])
			except Exception as e:
				raise Exception(
					f"Failed to initialize planner LLM with model 'o4-mini' for task agent. "
					f'This model may not be available or accessible in your project. '
					f'Check the Vertex AI Model Registry in the Google Cloud Console, or contact Google Cloud support to request access. '
					f'Error: {str(e)}'
				)

			fresh_agent = Agent(
				task=act,
				llm=exec_llm_task,
				planner_llm=plan_llm_task,
				planner_interval=1,
				browser=shared_browser,
				browser_context=shared_browser_ctx,
				close_browser_on_run=False,
				enable_memory=False,
			)

			# Set up screenshot handler
			task_screenshot_dir = os.path.join(task_run_dir, 'screenshots')
			os.makedirs(task_screenshot_dir, exist_ok=True)
			task_log_json = os.path.join(task_run_dir, 'log.json')
			task_handler = ScreenshotHandler(task_log_json, task_screenshot_dir)
			task_handler.setLevel(logging.INFO)
			for name in ['', 'browser_use', 'agent', 'controller']:
				logging.getLogger(name).addHandler(task_handler)

			# Log task start
			task_run_logger.info(f'Task Started: {act}')

			# Run the agent
			task_run_logger.info('Running Agent for Task')
			history = await fresh_agent.run()

			# Executor LLM Action Logging
			with open(task_executor_log, 'a', encoding='utf-8') as f:
				f.write('\n=== Executor Actions from History (Command Loop Task) ===\n')
				serialized_history = serialize_history(history)
				for i, entry in enumerate(serialized_history):
					f.write(f'\nStep {i + 1}:\n')
					if isinstance(entry, dict):
						if 'action' in entry:
							f.write(f'  Action: {json.dumps(entry["action"], indent=4, ensure_ascii=False)}\n')
						if 'llm_output' in entry:
							f.write(f'  LLM Output: {json.dumps(entry["llm_output"], indent=4, ensure_ascii=False)}\n')
						for key, value in entry.items():
							if key not in ['action', 'llm_output']:
								f.write(f'  {key}: {value}\n')
					else:
						f.write(f'  {entry}\n')
				f.write('\n')

			# Serialize and log history
			with open(os.path.join(task_run_dir, 'run_result.json'), 'w', encoding='utf-8') as f:
				json.dump(serialized_history, f, indent=2)

			# Log browser state
			try:
				# Access the browser from the context and get the current page
				browser = fresh_agent.browser
				if browser is not None:
					# Create a new context if none exists
					context = fresh_agent.browser_context
					if context is None:
						context = await browser.new_context()
					# Get the current page or create a new one
					page = await context.new_page() if not hasattr(context, 'pages') or not context.pages else context.pages[0]
					if page.url == 'about:blank':
						await page.goto('https://cloud.google.com')
					current_url = page.url
				else:
					current_url = 'unknown'
				task_run_logger.info(f'Browser State After Run: URL = {current_url}')
			except Exception as e:
				task_run_logger.error(f'Browser State Error: {str(e)}')
				current_url = 'unknown'

			# Force executor LLM call
			try:
				prompt = f"""
                Current URL: {current_url}
                Task: {act}
                Instructions: Verify the current page state and suggest the next action. Return JSON:
                {{
                    "current_state": {{"evaluation": "Description of page state"}},
                    "next_action": "Suggested action"
                }}
                """
				response = await exec_llm_task.ainvoke([HumanMessage(content=prompt)])
				with open(task_executor_log, 'a', encoding='utf-8') as f:
					f.write('\n=== Forced Executor LLM Call ===\n')
					f.write(f'Prompt: {prompt}\n')
					if hasattr(response, 'content'):
						f.write(f'Response: {response.content}\n')
					else:
						f.write('No response content\n')
					f.flush()
			except Exception as e:
				with open(task_executor_log, 'a', encoding='utf-8') as f:
					f.write(f'\n=== Forced Executor LLM Error: {str(e)} ===\n')
					f.flush()

			# Clean up
			for name in ['', 'browser_use', 'agent', 'controller']:
				logging.getLogger(name).removeHandler(task_handler)
			del fresh_agent

	finally:
		if shared_browser_ctx:
			await shared_browser_ctx.close()
		if shared_browser:
			await shared_browser.close()
		run_logger.removeHandler(run_file_handler)
		run_file_handler.close()


# -----------------------------------------------------------------
# 6.  Entry‑point
# -----------------------------------------------------------------
if __name__ == '__main__':
	asyncio.run(main())
