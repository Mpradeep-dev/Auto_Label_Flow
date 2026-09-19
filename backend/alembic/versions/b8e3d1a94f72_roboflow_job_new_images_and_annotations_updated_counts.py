"""roboflow job new_images_count / annotations_updated_count (export: honest new-vs-updated reporting)

Revision ID: b8e3d1a94f72
Revises: a4f1c8e2b567
Create Date: 2026-09-18 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b8e3d1a94f72'
down_revision: Union[str, None] = 'a4f1c8e2b567'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('roboflow_jobs', sa.Column('new_images_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column(
        'roboflow_jobs', sa.Column('annotations_updated_count', sa.Integer(), nullable=False, server_default='0')
    )
    op.alter_column('roboflow_jobs', 'new_images_count', server_default=None)
    op.alter_column('roboflow_jobs', 'annotations_updated_count', server_default=None)


def downgrade() -> None:
    op.drop_column('roboflow_jobs', 'annotations_updated_count')
    op.drop_column('roboflow_jobs', 'new_images_count')
