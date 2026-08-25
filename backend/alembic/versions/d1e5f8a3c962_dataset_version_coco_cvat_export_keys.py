"""dataset version coco/cvat-xml export storage keys

Revision ID: d1e5f8a3c962
Revises: c4f7a92e1b83
Create Date: 2026-08-25 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd1e5f8a3c962'
down_revision: Union[str, None] = 'c4f7a92e1b83'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('dataset_versions', sa.Column('coco_export_storage_key', sa.String(length=1000), nullable=True))
    op.add_column('dataset_versions', sa.Column('cvat_export_storage_key', sa.String(length=1000), nullable=True))


def downgrade() -> None:
    op.drop_column('dataset_versions', 'cvat_export_storage_key')
    op.drop_column('dataset_versions', 'coco_export_storage_key')
