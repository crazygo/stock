#!/usr/bin/env python3
"""Zero-dependency Cloudflare R2 / S3 client using Python standard library.

Provides high-performance listing, uploading, downloading, and diffing against
Cloudflare R2 without requiring external dependencies (like boto3 or rclone).
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "r2_storage.json"


def load_r2_config(config_path: Optional[Path] = None) -> Dict[str, str]:
    """Load credentials from config file with environment variable fallback."""
    cfg_file = config_path or DEFAULT_CONFIG_PATH
    data: Dict[str, Any] = {}
    if cfg_file.exists():
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
        except Exception as e:
            pass

    endpoint = os.environ.get("R2_ENDPOINT") or data.get("endpoint", "")
    bucket = os.environ.get("R2_BUCKET") or data.get("bucket", "market-data")
    region = os.environ.get("R2_REGION") or data.get("region", "auto")
    access_key = os.environ.get("R2_ACCESS_KEY_ID") or data.get("access_key_id", "")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY") or data.get("secret_access_key", "")

    if not endpoint or not access_key or not secret_key:
        raise ValueError(
            f"Missing R2 credentials. Please ensure {cfg_file} exists or set "
            "R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY in environment."
        )

    return {
        "endpoint": endpoint.rstrip("/"),
        "bucket": bucket,
        "region": region,
        "access_key_id": access_key,
        "secret_access_key": secret_key,
    }


class R2Client:
    """Client for Cloudflare R2 bucket operations via AWS SigV4."""

    def __init__(self, config: Optional[Dict[str, str]] = None, config_path: Optional[Path] = None):
        self.config = config or load_r2_config(config_path)
        self.endpoint = self.config["endpoint"]
        self.bucket = self.config["bucket"]
        self.region = self.config["region"]
        self.access_key = self.config["access_key_id"]
        self.secret_key = self.config["secret_access_key"]
        self.host = urllib.parse.urlparse(self.endpoint).netloc

    def _sign(self, key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    def _get_signing_key(self, date_stamp: str) -> bytes:
        k_date = self._sign(("AWS4" + self.secret_key).encode("utf-8"), date_stamp)
        k_region = self._sign(k_date, self.region)
        k_service = self._sign(k_region, "s3")
        return self._sign(k_service, "aws4_request")

    def request(
        self,
        method: str,
        path: str = "",
        query: str = "",
        body: bytes = b"",
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> urllib.request.addinfourl:
        """Execute an AWS SigV4 signed HTTP request to Cloudflare R2."""
        canonical_uri = f"/{self.bucket}/{path.lstrip('/')}" if path else f"/{self.bucket}"
        now = datetime.datetime.now(datetime.timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        payload_hash = hashlib.sha256(body).hexdigest()

        raw_headers = {
            "host": self.host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
        }
        if extra_headers:
            raw_headers.update(extra_headers)

        headers = {k.lower(): str(v) for k, v in raw_headers.items()}
        signed_headers_list = sorted(headers.keys())
        signed_headers = ";".join(signed_headers_list)
        canonical_headers = "".join(f"{k}:{headers[k]}\n" for k in signed_headers_list)

        canonical_querystring = query
        canonical_request = (
            f"{method}\n{canonical_uri}\n{canonical_querystring}\n"
            f"{canonical_headers}\n{signed_headers}\n{payload_hash}"
        )

        algorithm = "AWS4-HMAC-SHA256"
        credential_scope = f"{date_stamp}/{self.region}/s3/aws4_request"
        string_to_sign = (
            f"{algorithm}\n{amz_date}\n{credential_scope}\n"
            f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        )

        signing_key = self._get_signing_key(date_stamp)
        signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        auth_header = (
            f"{algorithm} Credential={self.access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        headers["Authorization"] = auth_header

        url = f"{self.endpoint}{canonical_uri}"
        if query:
            url += f"?{query}"

        req = urllib.request.Request(
            url,
            data=body if method in ("PUT", "POST") else None,
            headers=headers,
            method=method,
        )
        return urllib.request.urlopen(req, timeout=30)

    def _canonical_query(self, params: Dict[str, Any]) -> str:
        """Encode query params per AWS SigV4 (sorted by key, RFC 3986 encoding)."""
        items = []
        for k in sorted(params.keys()):
            v = params[k]
            k_enc = urllib.parse.quote(str(k), safe="")
            v_enc = urllib.parse.quote(str(v), safe="")
            items.append(f"{k_enc}={v_enc}")
        return "&".join(items)

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """List objects in bucket with optional prefix, supporting continuation."""
        objects = []
        continuation_token = None
        ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"

        while True:
            params: Dict[str, Any] = {"list-type": "2", "max-keys": str(max_keys)}
            if prefix:
                params["prefix"] = prefix
            if continuation_token:
                params["continuation-token"] = continuation_token

            query = self._canonical_query(params)
            try:
                with self.request("GET", "", query=query) as resp:
                    xml_content = resp.read()
                    root = ET.fromstring(xml_content)

                    for entry in root.findall(f".//{ns}Contents"):
                        key_node = entry.find(f"{ns}Key")
                        size_node = entry.find(f"{ns}Size")
                        modified_node = entry.find(f"{ns}LastModified")
                        etag_node = entry.find(f"{ns}ETag")

                        key = key_node.text if key_node is not None else ""
                        size = int(size_node.text) if size_node is not None and size_node.text else 0
                        modified = modified_node.text if modified_node is not None else ""
                        etag = (etag_node.text or "").strip('"')

                        objects.append({
                            "key": key,
                            "size": size,
                            "last_modified": modified,
                            "etag": etag,
                        })

                    is_truncated_node = root.find(f"{ns}IsTruncated")
                    if is_truncated_node is not None and is_truncated_node.text == "true":
                        next_token_node = root.find(f"{ns}NextContinuationToken")
                        if next_token_node is not None and next_token_node.text:
                            continuation_token = next_token_node.text
                            continue
                    break
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"R2 list_objects failed: HTTP {e.code} - {err_body}") from e

        return objects

    def head_object(self, key: str) -> Optional[Dict[str, Any]]:
        """Check if an object exists and retrieve metadata."""
        try:
            with self.request("HEAD", key) as resp:
                size = int(resp.headers.get("content-length", 0))
                etag = (resp.headers.get("etag") or "").strip('"')
                last_modified = resp.headers.get("last-modified", "")
                return {"key": key, "size": size, "etag": etag, "last_modified": last_modified}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"R2 head_object failed for {key}: HTTP {e.code} - {err_body}") from e

    def get_object(self, key: str, local_path: Path) -> int:
        """Download an object from R2 to local path."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = local_path.with_suffix(local_path.suffix + ".r2download")
        try:
            with self.request("GET", key) as resp, open(temp_path, "wb") as f:
                bytes_written = 0
                while chunk := resp.read(128 * 1024):
                    f.write(chunk)
                    bytes_written += len(chunk)
            temp_path.replace(local_path)
            return bytes_written
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def put_object(self, local_path: Path, key: str) -> int:
        """Upload a local file to R2."""
        if not local_path.is_file():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        file_bytes = local_path.read_bytes()
        content_type = "application/octet-stream"
        if local_path.name.endswith(".json"):
            content_type = "application/json"
        elif local_path.name.endswith(".json.gz"):
            content_type = "application/gzip"
        elif local_path.name.endswith(".parquet"):
            content_type = "application/vnd.apache.parquet"
        elif local_path.name.endswith(".csv"):
            content_type = "text/csv"

        extra_headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(file_bytes)),
        }

        with self.request("PUT", key, body=file_bytes, extra_headers=extra_headers) as resp:
            if resp.status not in (200, 201):
                raise RuntimeError(f"Upload failed: status {resp.status}")
            return len(file_bytes)

    def delete_object(self, key: str) -> bool:
        """Delete an object from R2."""
        try:
            with self.request("DELETE", key) as resp:
                return resp.status in (200, 204)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return True
            raise

    def diff(
        self,
        local_base_dir: Path,
        remote_prefix: str = "",
        allowed_extensions: Optional[Tuple[str, ...]] = (".json.gz", ".parquet", ".csv", ".json"),
    ) -> Dict[str, Any]:
        """Fast diff comparison between local directory and R2 bucket prefix."""
        # 1. Gather remote objects
        remote_objs = self.list_objects(prefix=remote_prefix)
        remote_map = {obj["key"]: obj for obj in remote_objs}

        # 2. Gather local files
        local_map: Dict[str, Dict[str, Any]] = {}
        if local_base_dir.exists():
            for p in local_base_dir.rglob("*"):
                if p.is_file():
                    if allowed_extensions and not any(p.name.endswith(ext) for ext in allowed_extensions):
                        continue
                    rel = p.relative_to(local_base_dir).as_posix()
                    key = f"{remote_prefix.rstrip('/')}/{rel}" if remote_prefix else rel
                    local_map[key] = {
                        "path": p,
                        "size": p.stat().st_size,
                        "mtime": p.stat().st_mtime,
                        "rel": rel,
                    }

        all_keys = sorted(set(remote_map.keys()) | set(local_map.keys()))
        only_local = []
        only_remote = []
        different = []
        identical = []

        for k in all_keys:
            in_loc = k in local_map
            in_rem = k in remote_map
            if in_loc and not in_rem:
                only_local.append({"key": k, "local_path": str(local_map[k]["path"]), "size": local_map[k]["size"]})
            elif in_rem and not in_loc:
                only_remote.append({"key": k, "size": remote_map[k]["size"], "last_modified": remote_map[k]["last_modified"]})
            else:
                loc_size = local_map[k]["size"]
                rem_size = remote_map[k]["size"]
                if loc_size != rem_size:
                    different.append({
                        "key": k,
                        "local_size": loc_size,
                        "remote_size": rem_size,
                        "local_path": str(local_map[k]["path"]),
                    })
                else:
                    identical.append({"key": k, "size": loc_size})

        return {
            "remote_prefix": remote_prefix,
            "local_dir": str(local_base_dir),
            "only_local": only_local,
            "only_remote": only_remote,
            "different": different,
            "identical": identical,
        }
