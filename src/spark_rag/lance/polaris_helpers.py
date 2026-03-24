"""Thin Polaris REST helpers for Lance table registration and discovery."""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

# Polaris 1.3.0 API paths (verified):
#   OAuth:          POST /api/catalog/v1/oauth/tokens
#   Management:     /api/management/v1/catalogs
#   Namespaces:     /api/catalog/v1/{catalog}/namespaces
#   Generic tables: /api/catalog/polaris/v1/{catalog}/namespaces/{ns}/generic-tables
#                   (realm prefix 'polaris' required for generic table operations)

_REALM = "polaris"


def get_token(endpoint: str, client_id: str, client_secret: str) -> str:
    """Get OAuth2 access token from Polaris."""
    resp = requests.post(
        f"{endpoint}/api/catalog/v1/oauth/tokens",
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "PRINCIPAL_ROLE:ALL",
        },
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    logger.debug("Polaris token acquired for client_id=%s", client_id)
    return token


def ensure_catalog(
    endpoint: str, token: str, catalog_name: str, s3_bucket: str,
) -> None:
    """Create catalog if it doesn't exist. Idempotent."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    # Check if exists
    resp = requests.get(
        f"{endpoint}/api/management/v1/catalogs/{catalog_name}",
        headers=headers,
    )
    if resp.status_code == 200:
        logger.debug("Catalog '%s' already exists", catalog_name)
        return

    resp = requests.post(
        f"{endpoint}/api/management/v1/catalogs",
        headers=headers,
        json={
            "catalog": {
                "name": catalog_name,
                "type": "INTERNAL",
                "properties": {
                    "default-base-location": f"s3://{s3_bucket}/",
                },
                "storageConfigInfo": {
                    "storageType": "S3",
                    "allowedLocations": [f"s3://{s3_bucket}/"],
                },
            },
        },
    )
    resp.raise_for_status()
    logger.info("Created Polaris catalog '%s'", catalog_name)


def ensure_namespace(
    endpoint: str, token: str, catalog_name: str, namespace: str,
) -> None:
    """Create namespace if it doesn't exist. Idempotent."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(
        f"{endpoint}/api/catalog/v1/{catalog_name}/namespaces",
        headers=headers,
        json={"namespace": [namespace]},
    )
    if resp.status_code == 409:
        logger.debug("Namespace '%s' already exists", namespace)
        return
    if resp.status_code == 200:
        logger.info("Created Polaris namespace '%s.%s'", catalog_name, namespace)
        return
    resp.raise_for_status()


def register_table(
    endpoint: str,
    token: str,
    catalog_name: str,
    namespace: str,
    table_name: str,
    s3_uri: str,
    properties: dict | None = None,
) -> None:
    """Register a Lance table as a generic table in Polaris. Replaces if exists."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    base_url = (
        f"{endpoint}/api/catalog/{_REALM}/v1/{catalog_name}"
        f"/namespaces/{namespace}/generic-tables"
    )

    # Delete existing (no update API in Polaris generic tables)
    requests.delete(f"{base_url}/{table_name}", headers=headers)

    resp = requests.post(
        base_url,
        headers=headers,
        json={
            "name": table_name,
            "format": "lance",
            "base-location": s3_uri,
            "properties": properties or {},
        },
    )
    resp.raise_for_status()
    logger.info("Registered Lance table '%s' in Polaris at %s", table_name, s3_uri)


def load_table(
    endpoint: str,
    token: str,
    catalog_name: str,
    namespace: str,
    table_name: str,
) -> dict:
    """Load Lance table metadata from Polaris. Returns table dict with base-location."""
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{endpoint}/api/catalog/{_REALM}/v1/{catalog_name}"
        f"/namespaces/{namespace}/generic-tables/{table_name}",
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()
    # Response is {"table": {...}} — unwrap
    return data.get("table", data)
