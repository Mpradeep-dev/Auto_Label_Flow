"""index training_jobs.status and roboflow_jobs.status

Revision ID: a1b2c3d4e5f6
Revises: f4a8c17d2e93
Create Date: 2026-08-26 00:00:00.000000

Audit finding DB-05: job-status columns weren't indexed on any of the three
job tables, despite status being the natural filter for "show me the
running/queued jobs" UI polling. inference_jobs.status also picked up
`index=True` alongside the BE-03 partial-unique-index model change
(e7c2a4f691b5_one_active_inference_job_per_dataset only added the partial
unique index itself, not this plain one) — included here too so the
migration chain actually matches the model metadata the test suite builds
schema from.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'f4a8c17d2e93'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_inference_jobs_status", "inference_jobs", ["status"])
    op.create_index("ix_training_jobs_status", "training_jobs", ["status"])
    op.create_index("ix_roboflow_jobs_status", "roboflow_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_roboflow_jobs_status", table_name="roboflow_jobs")
    op.drop_index("ix_training_jobs_status", table_name="training_jobs")
    op.drop_index("ix_inference_jobs_status", table_name="inference_jobs")
