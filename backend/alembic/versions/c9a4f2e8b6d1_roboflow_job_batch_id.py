"""roboflow job batch_id (import raw pull, narrow to one upload batch)

Revision ID: c9a4f2e8b6d1
Revises: b2d9f4a1c7e3
Create Date: 2026-09-02 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c9a4f2e8b6d1'
down_revision: Union[str, None] = 'b2d9f4a1c7e3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('roboflow_jobs', sa.Column('batch_id', sa.String(length=200), nullable=True))


def downgrade() -> None:
    op.drop_column('roboflow_jobs', 'batch_id')
