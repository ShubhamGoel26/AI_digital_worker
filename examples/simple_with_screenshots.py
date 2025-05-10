import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent

# Adjust sys.path to include project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Create a directory for screenshots (for future use)
screenshot_dir = 'screenshots'
os.makedirs(screenshot_dir, exist_ok=True)

# Task and LLM config
task = 'Find the founders of browser-use and draft them a short personalized message'
llm = ChatOpenAI(
	model='gpt-4o',
	temperature=0.0,
)

# Initialize agent
agent = Agent(task=task, llm=llm)


async def main():
	result = await agent.run()
	print(result)


if __name__ == '__main__':
	asyncio.run(main())
