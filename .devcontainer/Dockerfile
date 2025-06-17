FROM python:3.12-slim

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && mkdir -p /home/${USERNAME}/fl-dev \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && echo "source /venv/bin/activate" >> /etc/bash.bashrc \
    && echo 'export PATH="$HOME/.local/bin:$PATH"' >> /etc/bash.bashrc \
    && echo 'export PATH="$HOME/.local/bin:$PATH"' > /etc/profile.d/local-bin-path.sh

WORKDIR /home/${USERNAME}/fl-dev

USER ${USERNAME}