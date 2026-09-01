"""azure blob import: images.is_external + blob_import_jobs

Revision ID: b2d9f4a1c7e3
Revises: e91a6c3f7b28
Create Date: 2026-09-02 00:00:00.000000

Backs "Import from Azure Blob" — registering images that already live in
the app's blob container as `Image` rows *by reference*, no byte copy.
`images.is_external` marks those rows so deleting them never deletes the
underlying blob; `blob_import_jobs` is the pollable progress row for the
background walk (mirrors `roboflow_jobs`).
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2d9f4a1c7e3'
down_revision: Union[str, None] = 'e91a6c3f7b28'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'images',
        sa.Column('is_external', sa.Boolean(), nullable=False, server_default=sa.false()),
    )

    op.create_table(
        'blob_import_jobs',
        sa.Column('project_id', sa.UUID(), nullable=False),
        sa.Column(
            'status',
            sa.Enum('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', name='blob_import_job_status'),
            nullable=False,
        ),
        sa.Column('prefix', sa.String(length=1000), nullable=False),
        sa.Column('label_format', sa.String(length=20), nullable=False),
        sa.Column('dataset_name', sa.String(length=200), nullable=True),
        sa.Column('result_dataset_id', sa.UUID(), nullable=True),
        sa.Column('total_items', sa.Integer(), nullable=False),
        sa.Column('processed_items', sa.Integer(), nullable=False),
        sa.Column('celery_task_id', sa.String(length=200), nullable=True),
        sa.Column('error', sa.String(length=2000), nullable=True),
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['result_dataset_id'], ['datasets.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_blob_import_jobs_project_id'), 'blob_import_jobs', ['project_id'], unique=False)
    op.create_index(op.f('ix_blob_import_jobs_status'), 'blob_import_jobs', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_blob_import_jobs_status'), table_name='blob_import_jobs')
    op.drop_index(op.f('ix_blob_import_jobs_project_id'), table_name='blob_import_jobs')
    op.drop_table('blob_import_jobs')
    sa.Enum(name='blob_import_job_status').drop(op.get_bind(), checkfirst=True)

    op.drop_column('images', 'is_external')
