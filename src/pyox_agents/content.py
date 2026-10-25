"""Images and files (such as PDFs) to send in a prompt.

    await agent.run(["What is in this picture?", image("cat.png")])
    await agent.run(["Summarize this report", file("report.pdf")])

Local files and bytes are sent inline, base64-encoded in a `data:` URL.
"""

import base64
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

from pyox_agents.messages import FilePart, ImagePart

Source = str | Path | bytes
"""A file path, the bytes themselves, or (as a `str`) a web address."""


def image(source: Source, *, media_type: str | None = None) -> ImagePart:
    """An image from a path, bytes (give `media_type`, e.g. "image/png") or a web address.

    A web address is sent as is: the API downloads the image itself.
    """
    if _is_url(source):
        return {"type": "image", "url": str(source)}
    return {"type": "image", "url": _data_url(source, media_type)}


def file(source: Source, *, media_type: str | None = None, filename: str | None = None) -> FilePart:
    """A file, such as a PDF, from a path or bytes (give `media_type`, e.g. "application/pdf").

    OpenAI needs the file contents, so give a path or bytes. A web address is
    sent as is, for OpenAI-compatible servers that download files themselves.
    """
    if filename is None:
        filename = _filename(source, media_type)
    url = str(source) if _is_url(source) else _data_url(source, media_type)
    return {"type": "file", "url": url, "filename": filename}


def _is_url(source: Source) -> bool:
    return isinstance(source, str) and source.startswith(("https://", "http://", "data:"))


def _data_url(source: Source, media_type: str | None) -> str:
    if isinstance(source, bytes):
        data = source
    else:
        data = Path(source).read_bytes()
        media_type = media_type or mimetypes.guess_type(source)[0]
    if media_type is None:
        raise ValueError("Unknown media type: give it with media_type=, e.g. media_type='image/png'")
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def _filename(source: Source, media_type: str | None) -> str:
    if isinstance(source, Path) or (isinstance(source, str) and not _is_url(source)):
        return Path(source).name
    if isinstance(source, str) and source.startswith(("https://", "http://")):
        return Path(urlparse(source).path).name or "file"
    # Bytes or a `data:` URL: no name, only the media type.
    return "file" + (mimetypes.guess_extension(media_type or "") or "")
