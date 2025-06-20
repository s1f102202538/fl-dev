#!/bin/bash

sudo chown -R vscode:vscode ./.venv
uv sync
source ./.venv/bin/activate
