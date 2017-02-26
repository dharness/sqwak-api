"""empty message

Revision ID: 1b141794b27a
Revises: f6e8cc63a021
Create Date: 2017-02-25 21:08:22.094742

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1b141794b27a'
down_revision = 'f6e8cc63a021'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(u'ml_app_owner_id_fkey', 'ml_app', type_='foreignkey')
    op.create_foreign_key(None, 'ml_app', 'user', ['owner_id'], ['id'], ondelete='CASCADE')
    op.drop_constraint(u'ml_class_ml_app_id_fkey', 'ml_class', type_='foreignkey')
    op.create_foreign_key(None, 'ml_class', 'ml_app', ['ml_app_id'], ['id'], ondelete='CASCADE')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'ml_class', type_='foreignkey')
    op.create_foreign_key(u'ml_class_ml_app_id_fkey', 'ml_class', 'ml_app', ['ml_app_id'], ['id'])
    op.drop_constraint(None, 'ml_app', type_='foreignkey')
    op.create_foreign_key(u'ml_app_owner_id_fkey', 'ml_app', 'user', ['owner_id'], ['id'])
    # ### end Alembic commands ###
