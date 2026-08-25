"""roboflow job cancel status and unannotated-only flag

Revision ID: 8b3c1a0f4d21
Revises: 5a1f6c7d9e02
Create Date: 2026-08-25 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8b3c1a0f4d21'
down_revision: Union[str, None] = '5a1f6c7d9e02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Postgres 12+ allows ADD VALUE inside a transaction as long as the new
    # value isn't *used* in that same transaction — the ADD COLUMN below
    # doesn't reference it, so this is safe as one migration.
    op.execute("ALTER TYPE roboflow_job_status ADD VALUE IF NOT EXISTS 'CANCELLED'")
    op.add_column(
        'roboflow_jobs',
        sa.Column('unannotated_only', sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.alter_column('roboflow_jobs', 'unannotated_only', server_default=None)


def downgrade() -> None:
    op.drop_column('roboflow_jobs', 'unannotated_only')
    # Postgres has no ALTER TYPE ... DROP VALUE — leaving 'CANCELLED' in
    # the enum on downgrade is the accepted tradeoff (same as any other
    # additive enum-value migration).
