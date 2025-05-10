import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI


async def test_model():
	# Load environment variables from .env file in parent directory
	load_dotenv('../.env')

	# Retrieve required environment variables
	PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
	LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
	CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

	# Validate environment variables
	if not all([PROJECT_ID, LOCATION, CREDENTIALS_PATH]):
		print('Missing environment variables. Please check your .env file.')
		print(f'GOOGLE_CLOUD_PROJECT: {PROJECT_ID}')
		print(f'GOOGLE_CLOUD_LOCATION: {LOCATION}')
		print(f'GOOGLE_APPLICATION_CREDENTIALS: {CREDENTIALS_PATH}')
		return

	# Set the credentials environment variable
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH

	# Initialize the ChatVertexAI model
	try:
		model = ChatVertexAI(
			model_name='gemini-2.5-pro-preview-03-25',
			project=PROJECT_ID,
			location=LOCATION,
		)
		# Test with a simple prompt
		response = await model.ainvoke([HumanMessage(content='Hello, world!')])
		print('Model response:', response.content)
	except Exception as e:
		print(f'Error: {e}')


if __name__ == '__main__':
	asyncio.run(test_model())
