"""polygon annotation geometry and SEGMENTER model kind

Revision ID: 9f2b6d4e8a17
Revises: b7d4e91a3c58
Create Date: 2026-08-26 00:00:00.000000

Adds a `shape_type` discriminator + `points` geometry column to both
annotation tables (BBOX stays the default; POLYGON covers hand-drawn
polygons and SAM-derived masks alike, stored as a point ring). `x1..y2`
stay NOT NULL for every row — for POLYGON rows they hold the bounding box
of `points`, computed server-side, so every existing bbox consumer (quality
rules, SORT tracker, exporters, dataset-version pinning) keeps working
unmodified. Also adds ModelKind.SEGMENTER for registering SAM/SAM2
checkpoints through the existing model-registration flow.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9f2b6d4e8a17'
down_revision: Union[str, None] = 'b7d4e91a3c58'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    shape_type_enum = postgresql.ENUM("BBOX", "POLYGON", name="annotation_shape_type")
    shape_type_enum.create(op.get_bind())

    for table in ("annotation_events", "annotations"):
        op.add_column(
            table,
            sa.Column("shape_type", shape_type_enum, nullable=False, server_default="BBOX"),
        )
        op.alter_column(table, "shape_type", server_default=None)
        op.add_column(table, sa.Column("points", postgresql.JSON(), nullable=True))

    # Same same-transaction-safe pattern as 8b3c1a0f4d21: the ADD COLUMNs
    # above don't reference the new enum value, so this is safe alongside them.
    op.execute("ALTER TYPE model_kind ADD VALUE IF NOT EXISTS 'SEGMENTER'")


def downgrade() -> None:
    for table in ("annotation_events", "annotations"):
        op.drop_column(table, "points")
        op.drop_column(table, "shape_type")
    op.execute("DROP TYPE annotation_shape_type")
    # Postgres has no ALTER TYPE ... DROP VALUE — leaving 'SEGMENTER' in
    # model_kind on downgrade is the accepted tradeoff, same as 8b3c1a0f4d21.
