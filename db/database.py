from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from config.settings import settings


@contextmanager
def get_conn():
    """Connexion PostgreSQL : validee (commit) a la sortie du bloc, annulee (rollback) en cas d'erreur."""
    with psycopg.connect(settings.database_url, row_factory=dict_row, connect_timeout=5) as conn:
        yield conn
