"""training_job.modal_function_call_id (Modal provider support)

Revision ID: a2b3c4d5e6f7
Revises: c3f9a1e6d4b2
Create Date: 2026-08-26 00:00:00.000000

Stores the Modal FunctionCall object ID returned by .spawn(), used to
poll for training completion and download artifacts.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a2b3c4d5e6f7'
down_revision: Union[str, None] = 'c3f9a1e6d4b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "training_jobs",
        sa.Column("modal_function_call_id", sa.String(300), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("training_jobs", "modal_function_call_id")
