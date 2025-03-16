import wandb

# Initialize the W&B API
api = wandb.Api()

# Specify your project and entity (username or team)
project_name = "RPP src-tar mini_0 4_patches =932= train-vit_b_16"
entity_name = "takonoselidze-charles-university"

# Fetch the most recent runs from the project
runs = api.runs(f"{entity_name}/{project_name}")

# Sort runs by creation date and fetch the top 10 most recent ones
recent_runs = sorted(runs, key=lambda x: x.created_at, reverse=True)[:10]

# Loop through the most recent runs and print the metrics or other info
for run in recent_runs:
    print(f"Run ID: {run.id}")
    print(f"Name: {run.name}")
    print(f"Created At: {run.created_at}")
    print(f"Config: {run.config}")  # Configuration settings
    print(f"Summary: {run.summary}")  # Final results (e.g., metrics)
    print("\n")
