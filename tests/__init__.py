import os

IS_GITHUB_RUNNER = os.environ.get("IS_GITHUB_RUNNER", "false") == "true"
