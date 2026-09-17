"""roboflow job images_only (import, skip Roboflow's existing labels)

Revision ID: a3e7c1f9d245
Revises: c9a4f2e8b6d1
Create Date: 2026-09-17 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3e7c1f9d245'
down_revision: Union[str, None] = 'c9a4f2e8b6d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'roboflow_jobs',
        sa.Column('images_only', sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.alter_column('roboflow_jobs', 'images_only', server_default=None)


def downgrade() -> None:
    op.drop_column('roboflow_jobs', 'images_only')
