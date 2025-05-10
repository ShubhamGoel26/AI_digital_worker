import os

import vertexai
from langchain_google_vertexai import ChatVertexAI

# (dotenv will auto-load .env if you prefer that)
vertexai.init(
	project=os.getenv('GOOGLE_CLOUD_PROJECT'),
	location=os.getenv('GOOGLE_CLOUD_LOCATION'),
)

chat = ChatVertexAI(
	model_name='gemini-2.5-pro-preview-03-25',
	streaming=True,
	temperature=0.0,
)

resp = chat.predict_messages([{'role': 'user', 'content': 'Hello, Gemini!'}])
print(resp.content)
