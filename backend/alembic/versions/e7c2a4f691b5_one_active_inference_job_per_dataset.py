"""at most one active inference job per dataset

Revision ID: e7c2a4f691b5
Revises: d1e5f8a3c962
Create Date: 2026-08-25 00:00:00.000000

Audit finding BE-03: nothing stopped two inference jobs from running
concurrently against the same dataset — the app relies entirely on the
single-worker Celery queue as a deployment *convention*, not an enforced
invariant, and `run_inference_batch` does a non-atomic per-image
delete-then-recreate of AUTO annotations that two interleaved jobs can
corrupt. A plain application-level "is there already a QUEUED/RUNNING job"
check has its own race (two requests can both pass the check before either
commits), so this is enforced at the database level instead: a partial
unique index that only applies to non-terminal statuses, so any dataset can
have unlimited job *history* but never more than one job actually in
flight at a time. The write that loses the race gets a normal
IntegrityError, which the API layer (inference_jobs.py) turns into a 409.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e7c2a4f691b5'
down_revision: Union[str, None] = 'd1e5f8a3c962'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE UNIQUE INDEX uq_inference_jobs_one_active_per_dataset "
        "ON inference_jobs (dataset_id) "
        "WHERE status IN ('QUEUED', 'RUNNING')"
    )


def downgrade() -> None:
    op.execute("DROP INDEX uq_inference_jobs_one_active_per_dataset")
