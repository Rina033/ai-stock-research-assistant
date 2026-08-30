"""
Lakebase (Databricks-managed Postgres) connection helper.

Uses Lakebase OAuth V1 credential generation: the app's service principal
authenticates via the Databricks SDK and generates a short-lived token
that serves as the Postgres password. Falls back to a secret-based URL
if the OAuth flow is unavailable.
"""

import base64
import logging
import os
import time
from contextlib import contextmanager
from urllib.parse import urlparse, urlunparse

import psycopg2
from databricks.sdk import WorkspaceClient
from psycopg2.extras import RealDictCursor

log = logging.getLogger(__name__)
_w = WorkspaceClient()

_PROJECT = "projects/ai-stock-assistant/branches/production"
_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

# Token cache
_pg_password = None
_last_refresh = 0
_REFRESH_INTERVAL = 900  # 15 minutes
_cached_host = None
_cached_endpoint_name = None


def _get_endpoint() -> dict:
    """Return the configured Lakebase endpoint from the project branch."""
    resp = _w.api_client.do("GET", f"/api/2.0/postgres/{_PROJECT}/endpoints")
    endpoints = resp.get("endpoints", [])
    if not endpoints:
        raise RuntimeError(f"No Lakebase endpoint found under {_PROJECT}.")

    configured_host = os.environ.get("PGHOST", "").strip()
    if configured_host:
        for endpoint in endpoints:
            endpoint_host = endpoint.get("status", {}).get("hosts", {}).get("host", "")
            if endpoint_host == configured_host:
                return endpoint

    if len(endpoints) == 1:
        return endpoints[0]

    raise RuntimeError(
        "Multiple Lakebase endpoints found. Set PGHOST to the hostname of the endpoint to use."
    )


def _get_endpoint_host() -> str:
    """Fetch the current endpoint host from the Lakebase API."""
    global _cached_host
    if _cached_host:
        return _cached_host
    endpoint = _get_endpoint()
    host = endpoint.get("status", {}).get("hosts", {}).get("host", "")
    if not host:
        raise RuntimeError("The selected Lakebase endpoint did not return a hostname.")
    _cached_host = host
    return host


def _get_endpoint_name() -> str:
    """Return the full endpoint resource name required by the credential API."""
    global _cached_endpoint_name
    if _cached_endpoint_name:
        return _cached_endpoint_name
    configured_name = os.environ.get("LAKEBASE_ENDPOINT_NAME", "").strip()
    if configured_name:
        _cached_endpoint_name = configured_name
        return configured_name
    endpoint = _get_endpoint()
    name = endpoint.get("name", "")
    if not name:
        raise RuntimeError("The selected Lakebase endpoint did not return a resource name.")
    _cached_endpoint_name = name
    return name


def _get_password() -> str:
    """Generate a Lakebase OAuth credential token for the app's service principal."""
    global _pg_password, _last_refresh
    if _pg_password and time.time() - _last_refresh < _REFRESH_INTERVAL:
        return _pg_password

    endpoint_name = _get_endpoint_name()
    cred = _w.postgres.generate_database_credential(endpoint=endpoint_name)
    token = (cred.token or "").strip()
    if not token:
        raise RuntimeError("Lakebase credential API returned an empty token.")

    _pg_password = token
    _last_refresh = time.time()
    return _pg_password


def _get_connection_params() -> dict:
    """Build psycopg2 connection parameters using OAuth credential."""
    # Use PGHOST from env or fetch from API
    host = os.environ.get("PGHOST", "") or _get_endpoint_host()
    if not host:
        raise RuntimeError("PGHOST not set and could not fetch endpoint host from API.")

    return {
        "host": host,
        "database": os.environ.get("PGDATABASE", "databricks_postgres"),
        "user": os.environ.get("PGUSER", "").strip()
        or os.environ.get("DATABRICKS_CLIENT_ID", "").strip(),
        "port": os.environ.get("PGPORT", "5432"),
        "password": _get_password(),
        "sslmode": os.environ.get("PGSSLMODE", "require"),
        "cursor_factory": RealDictCursor,
    }


@contextmanager
def get_connection():
    """Yield a raw psycopg2 connection using Lakebase OAuth credential."""
    params = _get_connection_params()
    log.info("Connecting: host=%s db=%s user=%s", params["host"], params["database"], params["user"])
    conn = psycopg2.connect(**params)
    try:
        yield conn
    finally:
        conn.close()


def _fallback_secret_url() -> str:
    """Build a connection URL from the secret, replacing the host with the current endpoint."""
    secret = _w.secrets.get_secret(scope=_SCOPE, key=_KEY)
    url = base64.b64decode(secret.value).decode("utf-8")
    endpoint_host = _get_endpoint_host()
    if endpoint_host:
        parsed = urlparse(url)
        if parsed.hostname != endpoint_host:
            new_netloc = endpoint_host
            if parsed.username:
                new_netloc = f"{parsed.username}:{parsed.password}@{new_netloc}"
            url = urlunparse((parsed.scheme, new_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    return url


def get_engine():
    """Return a SQLAlchemy engine for Lakebase."""
    from sqlalchemy import create_engine
    return create_engine(_fallback_secret_url())


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE against Lakebase, return affected row count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount
