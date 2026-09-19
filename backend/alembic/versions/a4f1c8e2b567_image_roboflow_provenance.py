"""image roboflow provenance (image_id/workspace/project_slug) for annotation-only re-push

Revision ID: a4f1c8e2b567
Revises: d1e5a7c3f902
Create Date: 2026-09-18 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a4f1c8e2b567'
down_revision: Union[str, None] = 'd1e5a7c3f902'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('images', sa.Column('roboflow_image_id', sa.String(length=200), nullable=True))
    op.add_column('images', sa.Column('roboflow_workspace', sa.String(length=200), nullable=True))
    op.add_column('images', sa.Column('roboflow_project_slug', sa.String(length=200), nullable=True))


def downgrade() -> None:
    op.drop_column('images', 'roboflow_project_slug')
    op.drop_column('images', 'roboflow_workspace')
    op.drop_column('images', 'roboflow_image_id')
