"""File-scoped Drive access; callers persist reserved IDs before creating files."""

from __future__ import annotations
from typing import Any

import hashlib
from pathlib import Path


def checksum(path: Path) -> str:
    """Compute the local checksum used to verify Drive transfers."""
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def connect() -> Any:
    """Create a Drive client with file-scoped application credentials."""
    import google.auth
    from googleapiclient.discovery import build

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.file"])
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def children(service: Any, folder: Any) -> Any:
    """Yield every nontrashed child across Drive result pages."""
    if not folder or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in folder):
        raise ValueError("invalid Drive folder ID")
    token = None
    while True:
        page = service.files().list(q=f"'{folder}' in parents and trashed = false", pageToken=token,
                                    fields="nextPageToken,files(id,name,size,mimeType)", pageSize=1000).execute(num_retries=3)
        yield from page.get("files", [])
        token = page.get("nextPageToken")
        if not token:
            break


def reserve_id(service: Any) -> str:
    """Reserve a stable Drive ID before creating an artifact."""
    return service.files().generateIds(count=1, space="drive", type="files").execute(num_retries=3)["ids"][0]


def create(service: Any, file_id: str, name: str, parent: Any, *, folder: Any=False) -> Any:
    """Create a reserved artifact, reconciling ambiguous create responses."""
    metadata = {"id": file_id, "name": name, "parents": [parent]}
    if folder:
        metadata["mimeType"] = "application/vnd.google-apps.folder"
    try:
        return service.files().create(body=metadata, fields="id,name,parents,mimeType").execute()
    except Exception:
        # A lost create response must never cause allocation of a second ID.
        existing = service.files().get(fileId=file_id, fields="id,name,parents,mimeType").execute(num_retries=3)
        if existing["name"] != name or existing.get("parents") != [parent]:
            raise ValueError("reserved file ID belongs to another artifact")
        if folder and existing["mimeType"] != metadata["mimeType"]:
            raise ValueError("reserved folder ID is not a folder")
        return existing


def upload(service: Any, file_id: str, path: Path) -> Any:
    """Upload file bytes and verify remote size and checksum."""
    from googleapiclient.http import MediaFileUpload

    path = Path(path)
    media = MediaFileUpload(str(path), chunksize=8 * 1024 * 1024, resumable=True)
    request = service.files().update(fileId=file_id, media_body=media, fields="id,size,md5Checksum")
    response = None
    while response is None:
        _, response = request.next_chunk(num_retries=3)
    digest = checksum(path)
    if int(response["size"]) != path.stat().st_size or response["md5Checksum"] != digest:
        raise ValueError("Drive upload did not match the local artifact")
    return response


def download(service: Any, file_id: str, path: Path) -> None:
    """Download and verify bytes before replacing the destination."""
    from googleapiclient.http import MediaIoBaseDownload

    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    metadata = service.files().get(fileId=file_id, fields="id,size,md5Checksum").execute(num_retries=3)
    with partial.open("wb") as stream:
        request = MediaIoBaseDownload(stream, service.files().get_media(fileId=file_id), chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = request.next_chunk(num_retries=3)
    digest = checksum(partial)
    if partial.stat().st_size != int(metadata["size"]) or digest != metadata["md5Checksum"]:
        raise ValueError("Drive download is incomplete")
    partial.replace(path)
