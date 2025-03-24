import wandb

# Authenticate W&B API
# wandb.login()  # Or set WANDB_API_KEY in env variables

# Set your project details
ENTITY = "takonoselidze-charles-university"  # Replace with your W&B username or team name
api = wandb.Api()

# Get all projects under the entity
projects = api.projects(ENTITY)

for project in projects:
    project_name = project.name
    runs = api.runs(f"{ENTITY}/{project_name}")

    crashed_runs = [run for run in runs if run.state == "crashed"]

    if crashed_runs:
        print(f"Deleting {len(crashed_runs)} crashed runs in project: {project_name}")
        for run in crashed_runs:
            print(f"  - Deleting run: {run.id} - {run.name}")
            run.delete()

    # If the project has no other runs, delete the project
    remaining_runs = api.runs(f"{ENTITY}/{project_name}")  # Refresh runs list
    if len(remaining_runs) == 0:
        print(f"Deleting empty project: {project_name}")
        project.delete()