#!/bin/bash

if [ -f "./venv/bin/activate" ] && [ -f "./venv/pyvenv.cfg" ]; then
    sudo chown -R vscode:vscode ./venv
    source ./venv/bin/activate
else
    sudo chown -R vscode:vscode ./venv
    python -m pip install --upgrade pip
    python -m venv venv
    source ./venv/bin/activate
fi

pip install -r requirements.txt