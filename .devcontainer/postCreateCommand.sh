#!/bin/bash

sudo chown -R vscode:vscode ./venv
python -m pip install --upgrade pip
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt