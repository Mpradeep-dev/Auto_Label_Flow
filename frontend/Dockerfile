# syntax=docker/dockerfile:1
FROM node:22-slim

WORKDIR /app

# package-lock.json wasn't copied before this — `npm install` had no
# lockfile to read inside the container, so every build re-resolved the
# whole dependency tree from the registry instead of installing the exact,
# already-resolved versions. Copying it and switching to `npm ci` (which
# requires a lockfile) makes installs both deterministic and faster; the
# cache mount lets npm reuse downloaded packages across builds the same way
# backend/Dockerfile's pip cache mount does for torch/pip.
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
