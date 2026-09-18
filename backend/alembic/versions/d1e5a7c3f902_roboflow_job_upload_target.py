"""roboflow job upload_target (export: which Annotate-board column to push into)

Revision ID: d1e5a7c3f902
Revises: b7d24e5f9a13
Create Date: 2026-09-18 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd1e5a7c3f902'
down_revision: Union[str, None] = 'b7d24e5f9a13'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'roboflow_jobs',
        sa.Column('upload_target', sa.String(length=20), nullable=False, server_default='ANNOTATING'),
    )
    op.alter_column('roboflow_jobs', 'upload_target', server_default=None)


def downgrade() -> None:
    op.drop_column('roboflow_jobs', 'upload_target')
