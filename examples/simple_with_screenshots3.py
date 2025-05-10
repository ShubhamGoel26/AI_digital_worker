import asyncio
import json
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mss import mss

from browser_use import Agent

# Adjust sys.path to include project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Create directories for screenshots and debug logs
screenshot_dir = 'screenshots'
debug_log_dir = 'debug_logs'
os.makedirs(screenshot_dir, exist_ok=True)
os.makedirs(debug_log_dir, exist_ok=True)

# Task and LLM config
task = 'Find the founders of browser-use and draft them a short personalized message'
llm = ChatOpenAI(
	model='gpt-4o',
	temperature=0.0,
)


class ScreenshotHandler(logging.Handler):
	def __init__(self, log_file, screenshot_dir):
		super().__init__()
		self.log_file = log_file
		self.screenshot_dir = screenshot_dir
		self.step = 1
		self.last_goal = ''
		self.debug_file = os.path.join(debug_log_dir, 'debug_log.txt')

	def emit(self, record):
		message = record.getMessage()
		with open(self.debug_file, 'a', encoding='utf-8') as f:
			f.write(f"DEBUG: Checking message: '{message}'\n")
		triggers = ['Action 1/1:', 'Next goal:', 'Searched for']
		if any(trigger in message for trigger in triggers):
			timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
			filename = f'{self.screenshot_dir}/step_{self.step}_{timestamp}.png'
			try:
				with mss() as sct:
					sct.shot(output=filename)
				log_entry = {
					'step': self.step,
					'action': message if 'Action' in message or 'Searched for' in message else '',
					'next_goal': message if 'Next goal' in message else self.last_goal,
					'screenshot': filename,
				}
				with open(self.log_file, 'a', encoding='utf-8') as f:
					json.dump(log_entry, f)
					f.write('\n')
				print(f'Step {self.step}: Saved {filename}')
				if 'Next goal:' in message:
					self.last_goal = message
				self.step += 1
			except Exception as e:
				print(f'Failed to save screenshot: {e}')


async def main():
	# Set up logging handler
	log_file = f'{screenshot_dir}/log.json'
	handler = ScreenshotHandler(log_file, screenshot_dir)
	handler.setLevel(logging.INFO)
	# Attach to root logger and specific loggers
	logging.getLogger('').addHandler(handler)  # Root logger
	logging.getLogger('browser_use').addHandler(handler)
	logging.getLogger('agent').addHandler(handler)
	logging.getLogger('controller').addHandler(handler)

	# Initialize agent
	agent = Agent(task=task, llm=llm)

	try:
		result = await agent.run()
		print(result)
	finally:
		# Clean up handler
		logging.getLogger('').removeHandler(handler)
		logging.getLogger('browser_use').removeHandler(handler)
		logging.getLogger('agent').removeHandler(handler)
		logging.getLogger('controller').removeHandler(handler)


if __name__ == '__main__':
	asyncio.run(main())
