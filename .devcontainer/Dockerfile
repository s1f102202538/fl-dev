FROM python:3.12-slim

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

WORKDIR /fl-dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && mkdir -p /fl-dev/venv \
    && chown -R ${USER_UID}:${USER_GID} /fl-dev/venv \
    && python -m venv /fl-dev/venv \
    && echo "source /fl-dev/venv/bin/activate" >> /etc/bash.bashrc \
    && echo "export PATH=/fl-dev/venv/bin:$PATH" >> /etc/bash.bashrc

USER ${USERNAME}