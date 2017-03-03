"""empty message

Revision ID: 8cbf147fde06
Revises: 1b141794b27a
Create Date: 2017-03-02 19:00:48.083648

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8cbf147fde06'
down_revision = '1b141794b27a'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('audio_sample', 'label')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('audio_sample', sa.Column('label', sa.VARCHAR(), autoincrement=False, nullable=False))
    # ### end Alembic commands ###
