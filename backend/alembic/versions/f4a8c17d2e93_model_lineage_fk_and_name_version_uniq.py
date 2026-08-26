"""model lineage FK and name+version uniqueness

Revision ID: f4a8c17d2e93
Revises: e7c2a4f691b5
Create Date: 2026-08-26 00:00:00.000000

Two related audit findings on `models`:

  - DB-02: `base_model_id` was a bare UUID column with no FK constraint,
    unlike TrainingJob.base_model_id/result_model_id, which already
    reference this same table — could silently point at a deleted or
    nonexistent model. Backfills any currently-dangling value to NULL
    before adding the constraint, so an existing dirty value can't make
    the ALTER TABLE fail.
  - DB-03: no uniqueness on (name, version) meant registering the same
    model twice silently created two confusing rows. Made safe to add by
    a companion fix in workers/tasks/training.py that gives each retrain
    of the same base model a distinct version string.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f4a8c17d2e93'
down_revision: Union[str, None] = 'e7c2a4f691b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "UPDATE models SET base_model_id = NULL "
        "WHERE base_model_id IS NOT NULL AND base_model_id NOT IN (SELECT id FROM models)"
    )
    op.create_foreign_key(
        "fk_models_base_model_id_models",
        "models",
        "models",
        ["base_model_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_unique_constraint("uq_models_name_version", "models", ["name", "version"])


def downgrade() -> None:
    op.drop_constraint("uq_models_name_version", "models", type_="unique")
    op.drop_constraint("fk_models_base_model_id_models", "models", type_="foreignkey")
