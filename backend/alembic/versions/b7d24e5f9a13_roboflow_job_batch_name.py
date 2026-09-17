"""roboflow job batch_name (export: user-chosen upload batch label)

Revision ID: b7d24e5f9a13
Revises: a3e7c1f9d245
Create Date: 2026-09-17 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7d24e5f9a13'
down_revision: Union[str, None] = 'a3e7c1f9d245'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('roboflow_jobs', sa.Column('batch_name', sa.String(length=200), nullable=True))


def downgrade() -> None:
    op.drop_column('roboflow_jobs', 'batch_name')
