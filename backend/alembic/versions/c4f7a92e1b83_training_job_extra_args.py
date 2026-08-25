"""training job extra_args passthrough for arbitrary yolo train() kwargs

Revision ID: c4f7a92e1b83
Revises: 8b3c1a0f4d21
Create Date: 2026-08-25 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4f7a92e1b83'
down_revision: Union[str, None] = '8b3c1a0f4d21'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'training_jobs',
        sa.Column('extra_args', sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.alter_column('training_jobs', 'extra_args', server_default=None)


def downgrade() -> None:
    op.drop_column('training_jobs', 'extra_args')
