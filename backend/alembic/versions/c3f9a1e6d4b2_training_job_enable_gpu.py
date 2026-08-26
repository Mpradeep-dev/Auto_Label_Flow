"""training_job.enable_gpu (Kaggle GPU quota control)

Revision ID: c3f9a1e6d4b2
Revises: 9f2b6d4e8a17
Create Date: 2026-08-26 00:00:00.000000

Kaggle accounts have a weekly GPU-hours quota; the Kaggle training provider
previously requested a GPU kernel unconditionally with no way to run a
quota-free CPU kernel instead. Defaults true so every existing row keeps the
prior always-on behavior.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c3f9a1e6d4b2'
down_revision: Union[str, None] = '9f2b6d4e8a17'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "training_jobs",
        sa.Column("enable_gpu", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.alter_column("training_jobs", "enable_gpu", server_default=None)


def downgrade() -> None:
    op.drop_column("training_jobs", "enable_gpu")
