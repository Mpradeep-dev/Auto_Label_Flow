# Contributing

Thanks for considering a contribution to AutoLabelFlow.

## Getting set up

Follow the **Local development (without Docker)** section in
[`README.md`](README.md#local-development-without-docker) to get the
backend and frontend running. `docker compose up --build` also works if you
just want the whole stack running without editing it.

## Making a change

1. Fork the repo and create a branch off `main` for your change.
2. Keep the change focused — one fix or feature per PR.
3. Run the relevant tests before opening a PR:

   ```bash
   # backend
   cd backend
   DATABASE_URL=postgresql+psycopg://annotate:annotate@localhost:5432/annotate_test ./venv/Scripts/python -m pytest tests/ -q

   # frontend
   cd frontend
   npx tsc -b && npx vitest run
   ```
4. Open a pull request against `main` describing what changed and why.

## Reporting bugs / requesting features

- **Bugs**: open an [issue](https://github.com/Mpradeep-dev/Auto_Label_Flow/issues/new)
  with steps to reproduce, what you expected, and what happened instead
  (include logs/screenshots if relevant).
- **Feature ideas / questions**: use
  [discussions](https://github.com/Mpradeep-dev/Auto_Label_Flow/discussions)
  rather than an issue, so we can talk through the idea before it becomes a
  concrete task.

## Code style

- Backend: FastAPI/Pydantic v2/SQLAlchemy 2.x conventions already in
  `backend/app/`. Follow the existing service-layer interfaces
  (`ObjectStorage`, `TrainingProvider`, quality-rule packs) rather than
  hardcoding new one-off logic.
- Frontend: React/TypeScript, Swiss International design system — see
  [`frontend/DESIGN.md`](frontend/DESIGN.md) before adding UI.
