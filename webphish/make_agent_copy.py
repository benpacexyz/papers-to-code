##########
# ### Import Packages and Setup Paths

# standard
import os
import sys
import shutil

# __file__ not defined (e.g. PyCharm console)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# setup agent directory next to this repo
AGENT_DIR = os.path.join(SCRIPT_DIR, '..', 'webphish-auto')

##########
# ### Copy Agent Files

# files given to the agent, data/raw and html_valid stay behind
agent_files = [
    'pyproject.toml',
    '.python-version',
    '.gitignore',
    '__init__.py',
    'src/__init__.py',
    'src/utils.py',
    'src/step_00_train_test_split.py',
    'src/step_01_training.py',
    'paper/2024_Expert Systems with Applications_Look Before You Leap.pdf',
    'data/train-test/html_train.parquet',
    'data/train-test/html_test.parquet',
]

# do not build over a previous attempt
if os.path.exists(AGENT_DIR):
    sys.exit('webphish-auto already exists, run "make agent-clean" first')

# copy the agent files
for file_path in agent_files:
    os.makedirs(os.path.join(AGENT_DIR, os.path.dirname(file_path)), exist_ok=True)
    shutil.copy(
        os.path.join(SCRIPT_DIR, file_path),
        os.path.join(AGENT_DIR, file_path)
    )

# rename the project for the agent copy
with open(os.path.join(AGENT_DIR, 'pyproject.toml'), 'r') as f:
    pyproject_text = f.read()
with open(os.path.join(AGENT_DIR, 'pyproject.toml'), 'w') as f:
    f.write(pyproject_text.replace('name = "webphish"', 'name = "webphish-auto"'))

# add an empty scratch folder for the agent
os.makedirs(os.path.join(AGENT_DIR, 'agent'))
