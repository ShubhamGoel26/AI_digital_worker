Improving VLM Instruct Models for Cloud Console Automation

Banner image illustrating cloud console automation with VLMs.
This project enhances Vision-Language Models (VLMs), specifically Qwen-2.5 VL-Chat, to navigate and perform tasks in cloud management consoles, with a focus on Google Cloud Platform (GCP). We introduce a novel dataset of 10,061 screenshot-instruction-action triples and demonstrate that our VLM completes 83% of 236 GCP workflows with ≤20 prompts. Developed as part of the DS-GA 1012 Natural Language Understanding course at NYU Center for Data Science, this work addresses the unique challenges of cloud consoles—nested menus, dynamic DOM elements, and strict security policies—paving the way for reliable GUI automation.

Key Contributions

Dataset Creation: Compiled 10,061 multimodal triples across 13 GCP services, the first public corpus for cloud-console automation.
Persistent Session Hack: Modified the browser-use agent to maintain login state, enabling seamless automation after a one-time human login.
Prompt-Efficiency Study: Showed most tasks are solved with ≤20 prompts, with detailed distributions for Planner and Executor LLMs.


Why Cloud Consoles Are Hard

Nested Menus & Dynamic DOM: Asynchronous UI elements challenge naive agents.
Security Policies: MFA and device checks disrupt session persistence.
Lack of Data: No prior datasets paired real console screenshots with precise actions—until now.


Pipeline & Architecture
A Planner LLM decomposes user goals into sub-tasks, an Executor LLM generates GUI actions, and a persistent browser agent executes them, logging every step. Key modules include:

Planner: Breaks down tasks into actionable steps (e.g., "Click 'Create Bucket'").
Executor: Emits low-latency commands based on screenshots.
Persistent Browser: Retains session state using close_browser_on_run=False.
Logger: Captures screenshots, prompts, and actions.
State Tracker: Detects loops via URL/DOM hashing.
Recovery Manager: Handles retries and waits for dynamic elements.


Dataset Snapshot

Size: 10,061 triples across 236 GCP workflows.
Format: JSONL with {image_path, instruction, action} entries.
Sample:{
  "image_path": "screenshots/step_1_2023-04-25.png",
  "instruction": "Click on the 'Create Bucket' button",
  "action": "click_element(index=5)"
}


Examples: Permissions, role filtering, account setup (see poster visuals).


Prompt-Count Distributions
Both Planner and Executor LLMs excel with low prompt counts:

Planner: Majority of tasks use 5-19 prompts (Table 1, report).
Executor: Most actions completed in 5-19 prompts (Table 2, report).

Figure: Prompt-count distributions for Planner and Executor LLMs.

Setup Instructions

Clone the Repository:git clone https://github.com/dpraj07/Browser-use-persistant-state.git


Install Dependencies:pip install -r requirements.txt


Dataset Access: Find the dataset in runs/ or export it via export_dataset.py.


Note: Requires browser-use and a Chromium browser.


Usage Examples

Run a Workflow:python gcp_administrator_agent.py --task "Navigate to GCP Home page"


Export Dataset:python export_dataset.py --input runs/ --output data/dataset.jsonl




Results

Success Rate: 83% of 236 GCP workflows completed in ≤20 prompts.
Efficiency: Prompt-count analysis confirms minimal steps for most tasks.
Applications: Handles tasks like provisioning VMs and configuring IAM roles.


Limitations & Future Work

Scrolling: Struggles with scroll-heavy pages; plan to enhance viewport heuristics.
Scope: Limited to GCP; future expansion to AWS/Azure planned.
Optimization: Reinforcement tuning aims to cut prompt counts by 30%.
Dynamic UIs: Targeting better handling of pop-ups and asynchronous updates.

Report issues via GitHub Issues.


Resources

Report: team_30_final_report.pdf
Poster: Team_30_poster.pdf

