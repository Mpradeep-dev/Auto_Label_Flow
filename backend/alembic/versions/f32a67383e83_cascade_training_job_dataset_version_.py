"""cascade training job dataset version delete

Revision ID: f32a67383e83
Revises: 27db89ec19c1
Create Date: 2026-08-25 01:09:26.172483

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f32a67383e83'
down_revision: Union[str, None] = '27db89ec19c1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint('training_jobs_dataset_version_id_fkey', 'training_jobs', type_='foreignkey')
    op.create_foreign_key(
        'training_jobs_dataset_version_id_fkey',
        'training_jobs',
        'dataset_versions',
        ['dataset_version_id'],
        ['id'],
        ondelete='CASCADE',
    )


def downgrade() -> None:
    # Named explicitly on both sides (autogenerate's `None` here produces a
    # constraint Alembic can't later DROP by name — it compiles with no
    # name to reference) — confirmed live: the generated downgrade() raised
    # `CompileError: Can't emit DROP CONSTRAINT ... it has no name`.
    op.drop_constraint('training_jobs_dataset_version_id_fkey', 'training_jobs', type_='foreignkey')
    op.create_foreign_key(
        'training_jobs_dataset_version_id_fkey',
        'training_jobs',
        'dataset_versions',
        ['dataset_version_id'],
        ['id'],
        ondelete='RESTRICT',
    )
