"""add models.is_promptable for open-vocabulary (YOLO-World) detectors

Revision ID: b7d4e91a3c58
Revises: a1b2c3d4e5f6
Create Date: 2026-08-26 00:00:00.000000

Marks a registered model as open-vocabulary: its `class_config` is only a
display fallback (the checkpoint's default vocabulary), not authoritative —
the real classes are supplied at inference time via
DetectionModel.set_classes(), see services/inference/registry.py. Defaults
false so every existing (closed-vocabulary) model row is unaffected.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b7d4e91a3c58'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "models",
        sa.Column("is_promptable", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    op.drop_column("models", "is_promptable")
