import asyncio
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from playwright.async_api import async_playwright

from browser_use import Agent

# Adjust sys.path to include project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Create a directory for screenshots
screenshot_dir = 'screenshots'
os.makedirs(screenshot_dir, exist_ok=True)

# Task and LLM config
task = 'Find the founders of browser-use and draft them a short personalized message'
llm = ChatOpenAI(
	model='gpt-4o',
	temperature=0.0,
)


async def main():
	async with async_playwright() as p:
		browser = await p.chromium.launch(headless=False)  # Visible for debugging
		page = await browser.new_page()
		agent = Agent(task=task, llm=llm)

		# Track steps
		step = 1
		log_file = f'{screenshot_dir}/log.json'

		def custom_print(*args, **kwargs):
			nonlocal step
			message = ' '.join(map(str, args))
			original_print(f'DEBUG: {message}')  # Debug log
			original_print(*args, **kwargs)
			# Broader condition to catch actions and goals
			if any(x in message for x in ['Action 1/1:', 'Next goal:', 'controller', 'Searched for']):
				timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
				filename = f'{screenshot_dir}/step_{step}_{timestamp}.png'
				try:
					asyncio.create_task(page.screenshot(path=filename))
					log_entry = {
						'step': step,
						'action': message if 'Action' in message or 'controller' in message else '',
						'next_goal': message if 'Next goal' in message else '',
						'screenshot': filename,
					}
					with open(log_file, 'a') as f:
						json.dump(log_entry, f)
						f.write('\n')
					original_print(f'Step {step}: Saved {filename}')
					step += 1
				except Exception as e:
					original_print(f'Failed to save screenshot: {e}')

		original_print = print
		import builtins

		builtins.print = custom_print

		try:
			result = await agent.run()
			original_print(result)
		finally:
			builtins.print = original_print
			await browser.close()


if __name__ == '__main__':
	asyncio.run(main())
