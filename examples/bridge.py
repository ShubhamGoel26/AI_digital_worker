# bridge.py
"""
Run with:  uvicorn bridge:app --reload
"""

import importlib
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ------------------------------------------------------------------
# 1.  Load the original backend as a module
# ------------------------------------------------------------------
backend = importlib.import_module('gcp_manual_login_persistent_new3')

app = FastAPI(title='GCP Agent Bridge', description='Tiny REST shim that turns console I/O into API calls.')
# 1️⃣  Serve the static folder
app.mount('/static', StaticFiles(directory='static'), name='static')


# 2️⃣  Serve index.html at the root /
@app.get('/', include_in_schema=False)
async def root():
	return FileResponse('static/index.html')


# ------------------------------------------------------------------
# 2.  Keep the shared Playwright browser alive across requests
# ------------------------------------------------------------------
class _Shared:
	browser: Optional[object] = None
	ctx: Optional[object] = None


SHARED = _Shared()


# ------------------------------------------------------------------
# 3.  Pydantic schema for /action
# ------------------------------------------------------------------
class Action(BaseModel):
	task: str


# ------------------------------------------------------------------
# 4‑A  Kick‑off: open GCP sign‑in page and wait for manual login
# ------------------------------------------------------------------
@app.post('/login')
async def login(background_tasks: BackgroundTasks):
	if SHARED.browser:  # already logged‑in
		return {'detail': 'Session already initialised'}

	async def _worker():
		first_agent = backend.Agent(
			task='Navigate to Google Cloud Platform and go to sign in',
			llm=backend.exec_llm,
			planner_llm=backend.plan_llm,
			planner_interval=1,
			use_vision_for_planner=False,
			is_planner_reasoning=False,
			close_browser_on_run=False,
			enable_memory=False,
		)
		await first_agent.run()  # opens browser
		print('[bridge]  Waiting for manual login in opened window …')
		# ------------------------------------------------------------------
		# Let the user finish login in the visible browser tab, then press
		# the “   Done   ” button on the web UI (JS calls /confirm_login).
		# ------------------------------------------------------------------
		SHARED.browser = first_agent.browser
		SHARED.ctx = first_agent.browser_context

	background_tasks.add_task(_worker)
	return {'detail': 'Browser launched – log in manually, then press ‘Done’'}


# ------------------------------------------------------------------
# 4‑B  Confirm that the user has finished logging in
# ------------------------------------------------------------------
@app.post('/confirm_login')
async def confirm():
	if not SHARED.browser:
		raise HTTPException(status_code=400, detail='Login not started yet')
	return {'detail': 'Login confirmed – ready to accept tasks'}


# ------------------------------------------------------------------
# 4‑C  Run arbitrary GCP tasks (was console `input()` loop)
# ------------------------------------------------------------------
@app.post('/action')
async def action(action: Action, background_tasks: BackgroundTasks):
	if not SHARED.browser:
		raise HTTPException(status_code=400, detail='Please log in first')

	async def _worker():
		agent = backend.Agent(
			task=action.task,
			llm=backend.exec_llm,
			planner_llm=backend.plan_llm,
			planner_interval=1,
			is_planner_reasoning=False,
			browser=SHARED.browser,
			browser_context=SHARED.ctx,
			close_browser_on_run=False,
			enable_memory=False,
		)
		await agent.run()

	background_tasks.add_task(_worker)
	return {'detail': f'Started: {action.task}'}


# ------------------------------------------------------------------
# 5.  Graceful shutdown
# ------------------------------------------------------------------
@app.on_event('shutdown')
async def _shutdown():
	if SHARED.ctx:
		await SHARED.ctx.close()
	if SHARED.browser:
		await SHARED.browser.close()
