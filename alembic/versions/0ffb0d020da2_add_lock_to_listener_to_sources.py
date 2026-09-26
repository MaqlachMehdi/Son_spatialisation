"""add lock_to_listener to sources

Revision ID: 0ffb0d020da2
Revises: 8f2c1a9d4b6e
Create Date: 2026-08-20 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0ffb0d020da2'
down_revision: Union[str, Sequence[str], None] = '8f2c1a9d4b6e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'sources',
        sa.Column('lock_to_listener', sa.Boolean(), nullable=False, server_default='true'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('sources', 'lock_to_listener')
