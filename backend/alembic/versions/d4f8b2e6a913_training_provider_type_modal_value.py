"""training_provider_type enum: add MODAL value

Revision ID: d4f8b2e6a913
Revises: a2b3c4d5e6f7
Create Date: 2026-08-27 00:00:00.000000

a2b3c4d5e6f7 added `training_jobs.modal_function_call_id` and the Python
`TrainingProviderType.MODAL` member, but never altered the Postgres
`training_provider_type` enum itself to accept the value — every query
filtering `provider == MODAL` (e.g. `poll_modal_training_jobs`, run every
120s by Celery Beat) has been failing with `InvalidTextRepresentation:
invalid input value for enum training_provider_type: "MODAL"` since that
migration, since no row can ever legitimately contain 'MODAL' for Postgres
to compare against. Same fix pattern as 9f2b6d4e8a17's `model_kind` value.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd4f8b2e6a913'
down_revision: Union[str, None] = 'a2b3c4d5e6f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE training_provider_type ADD VALUE IF NOT EXISTS 'MODAL'")


def downgrade() -> None:
    # Postgres has no ALTER TYPE ... DROP VALUE — leaving 'MODAL' in
    # training_provider_type on downgrade is the accepted tradeoff, same as
    # 9f2b6d4e8a17's model_kind.
    pass
