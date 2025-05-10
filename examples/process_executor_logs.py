import asyncio
import glob
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI


async def process_log_files():
	# Load environment variables from .env file in parent directory
	if not os.path.exists('../.env'):
		print('Error: .env file not found at C:/Users/Dell/browser-use/.env')
		return
	load_dotenv('../.env')

	# Retrieve environment variables
	PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
	LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
	CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

	# Validate required environment variables
	if not PROJECT_ID:
		print('Error: GOOGLE_CLOUD_PROJECT not found in .env file.')
		return
	if not LOCATION:
		print('Error: GOOGLE_CLOUD_LOCATION not found in .env file.')
		return
	if not CREDENTIALS_PATH:
		print('Error: GOOGLE_APPLICATION_CREDENTIALS not found in .env file.')
		return
	if not os.path.isfile(CREDENTIALS_PATH):
		print(f'Error: Service account key file at {CREDENTIALS_PATH} does not exist.')
		return

	# Set GOOGLE_APPLICATION_CREDENTIALS environment variable
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH

	# Initialize ChatVertexAI model
	try:
		model = ChatVertexAI(
			model_name='gemini-2.5-flash-preview-04-17',
			project=PROJECT_ID,
			location=LOCATION,
			temperature=0.5,
			max_retries=6,
		)
	except Exception as e:
		print(f'Error initializing ChatVertexAI: {e}')
		return

	# Find all executor_log.txt files
	log_files = glob.glob('runs/Gemini_new_dependent/dependent_task_*/executor_log.txt')
	if not log_files:
		print('Error: No executor_log.txt files found in runs/Gemini/task_*/')
		return

	print(f'Found {len(log_files)} log files to process.')

	for log_file in log_files:
		print(f'Processing file: {log_file}')
		task_folder = os.path.dirname(log_file)

		# Read the log file with UTF-8 encoding
		try:
			with open(log_file, 'r', encoding='utf-8') as f:
				lines = f.readlines()
		except UnicodeDecodeError as e:
			print(f'Error reading {log_file}: {e}. Skipping this file.')
			continue
		except Exception as e:
			print(f'Unexpected error reading {log_file}: {e}. Skipping this file.')
			continue

		# Find all prompt sections
		sections = []
		for i in range(len(lines)):
			if lines[i].strip() == '=== LLM Prompt (model=gemini-2.5-flash-preview-04-17) ===':
				start = i + 1
				for j in range(start, len(lines)):
					if lines[j].strip().startswith('==='):
						end = j
						break
				else:
					end = len(lines)
				sections.append((start, end))

		print(f'Found {len(sections)} prompt sections in {log_file}')

		# Process each prompt section
		offset = 0
		for start, end in sections:
			actual_start = start + offset
			actual_end = end + offset
			prompt_lines = lines[actual_start:actual_end]
			prompt = ''.join(prompt_lines).strip()

			# Use ChatVertexAI to generate response
			try:
				response = await model.ainvoke([HumanMessage(content=prompt)])
				llm_response = response.content
			except Exception as e:
				print(f'Error generating response for prompt in {log_file}: {e}. Skipping this prompt.')
				continue

			# Prepare response lines
			response_header = '=== LLM Response ==='
			response_content_lines = llm_response.split('\n')
			response_lines = [response_header + '\n'] + [line + '\n' for line in response_content_lines]

			# Insert response into the log file
			insert_pos = actual_end
			lines[insert_pos:insert_pos] = response_lines
			inserted = len(response_lines)
			offset += inserted

		# Save the modified log file
		processed_folder = os.path.join(task_folder, 'processed')
		os.makedirs(processed_folder, exist_ok=True)
		processed_file = os.path.join(processed_folder, 'executor_log_with_response.txt')
		try:
			with open(processed_file, 'w', encoding='utf-8') as f:
				f.writelines(lines)
			print(f'Saved processed file: {processed_file}')
		except Exception as e:
			print(f'Error writing to {processed_file}: {e}. Skipping save for this file.')
			continue


if __name__ == '__main__':
	asyncio.run(process_log_files())
