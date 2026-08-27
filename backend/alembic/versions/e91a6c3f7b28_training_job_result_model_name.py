"""training_job.result_model_name

Revision ID: e91a6c3f7b28
Revises: d4f8b2e6a913
Create Date: 2026-08-27 00:10:00.000000

Lets the caller of POST /training/jobs name the model a run will produce,
instead of every provider always hardcoding f"{base_model.name}-retrained"
(see training_job.py's own comment on this column). NULL preserves that
exact prior default at finalize time — purely additive, no data migration.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e91a6c3f7b28'
down_revision: Union[str, None] = 'd4f8b2e6a913'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "training_jobs",
        sa.Column("result_model_name", sa.String(200), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("training_jobs", "result_model_name")
