"""add roboflow jobs table

Revision ID: 5a1f6c7d9e02
Revises: f32a67383e83
Create Date: 2026-08-25 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5a1f6c7d9e02'
down_revision: Union[str, None] = 'f32a67383e83'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('roboflow_jobs',
    sa.Column('project_id', sa.UUID(), nullable=False),
    sa.Column('kind', sa.Enum('IMPORT', 'EXPORT', name='roboflow_job_kind'), nullable=False),
    sa.Column('status', sa.Enum('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', name='roboflow_job_status'), nullable=False),
    sa.Column('workspace', sa.String(length=200), nullable=False),
    sa.Column('project_slug', sa.String(length=200), nullable=False),
    sa.Column('version', sa.Integer(), nullable=True),
    sa.Column('dataset_name', sa.String(length=200), nullable=True),
    sa.Column('result_dataset_id', sa.UUID(), nullable=True),
    sa.Column('dataset_version_id', sa.UUID(), nullable=True),
    sa.Column('uploaded_count', sa.Integer(), nullable=False),
    sa.Column('failed_count', sa.Integer(), nullable=False),
    sa.Column('failures', sa.JSON(), nullable=False),
    sa.Column('total_items', sa.Integer(), nullable=False),
    sa.Column('processed_items', sa.Integer(), nullable=False),
    sa.Column('celery_task_id', sa.String(length=200), nullable=True),
    sa.Column('error', sa.String(length=2000), nullable=True),
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['dataset_version_id'], ['dataset_versions.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['result_dataset_id'], ['datasets.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_roboflow_jobs_project_id'), 'roboflow_jobs', ['project_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_roboflow_jobs_project_id'), table_name='roboflow_jobs')
    op.drop_table('roboflow_jobs')
    sa.Enum(name='roboflow_job_kind').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='roboflow_job_status').drop(op.get_bind(), checkfirst=True)
