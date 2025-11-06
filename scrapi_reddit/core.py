"""Core scraping utilities for the Scrapi Reddit toolkit."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import re
import sys
import time
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from html import unescape
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, List
from urllib.parse import urlparse, urlunparse

import requests

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
BASE_URL = "https://www.reddit.com"
LISTING_PAGE_SIZE = 100
MAX_COMMENT_LIMIT = 500
MIN_DELAY_SECONDS = 1.0

MEDIA_EXTENSION_WHITELIST = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".gifv",
    ".webp",
    ".bmp",
    ".tiff",
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
}

ANIMATED_IMAGE_EXTENSIONS = {
    ".gif",
}

STATIC_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
}

MEDIA_FILTER_CATEGORIES = {"video", "image", "animated", "audio"}

CONTENT_TYPE_EXTENSION_MAP = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
}


@dataclass(slots=True)
class ScrapeOptions:
    """Configuration shared by subreddit processing routines."""

    output_root: Path
    listing_limit: int | None
    comment_limit: int
    delay: float
    time_filter: str
    output_formats: set[str] = field(default_factory=lambda: {"json"})
    listing_page_size: int = LISTING_PAGE_SIZE
    fetch_comments: bool = False
    resume: bool = False
    download_media: bool = False
    media_filters: set[str] | None = None

    def __post_init__(self) -> None:
        if self.listing_limit is not None:
            if self.listing_limit <= 0:
                self.listing_limit = None
            else:
                self.listing_limit = max(1, self.listing_limit)
        if self.comment_limit <= 0:
            self.comment_limit = MAX_COMMENT_LIMIT
        else:
            self.comment_limit = max(1, min(self.comment_limit, MAX_COMMENT_LIMIT))
        self.delay = max(self.delay, MIN_DELAY_SECONDS)
        self.output_formats = set(self.output_formats)
        if self.time_filter not in {"hour", "day", "week", "month", "year", "all"}:
            raise ValueError(f"Unsupported time filter: {self.time_filter}")
        if not self.output_formats:
            raise ValueError("At least one output format is required")
        allowed_formats = {"json", "csv"}
        if not self.output_formats.issubset(allowed_formats):
            raise ValueError(f"Unsupported output formats: {self.output_formats}")
        self.listing_page_size = max(1, min(int(self.listing_page_size), LISTING_PAGE_SIZE))
        if self.media_filters is not None and not isinstance(self.media_filters, set):
            self.media_filters = set(self.media_filters)
        if self.media_filters:
            self.media_filters = normalize_media_filter_tokens(self.media_filters)


@dataclass(slots=True)
class ListingTarget:
    """Represents a single listing JSON endpoint to scrape."""

    label: str
    output_segments: tuple[str, ...]
    url: str
    params: dict[str, Any] = field(default_factory=dict)
    context: str = ""
    allow_limit: bool = True

    def output_dir(self, root: Path) -> Path:
        return root.joinpath(*self.output_segments)

    def resolved_context(self) -> str:
        return self.context or self.label


@dataclass(slots=True)
class PostTarget:
    """Represents a single post JSON endpoint to scrape."""

    label: str
    output_segments: tuple[str, ...]
    url: str
    params: dict[str, Any] = field(default_factory=dict)
    context: str = ""

    def output_dir(self, root: Path) -> Path:
        return root.joinpath(*self.output_segments)


def _determine_page_limit(
    *,
    target: ListingTarget,
    options: ScrapeOptions,
    fetched_count: int,
) -> int | None:
    if not target.allow_limit:
        # Respect any explicit limit already present but don't override with defaults.
        explicit = target.params.get("limit")
        if explicit is None:
            return None
        try:
            return max(1, min(int(explicit), LISTING_PAGE_SIZE))
        except (TypeError, ValueError):
            return LISTING_PAGE_SIZE

    explicit_limit = target.params.get("limit")
    page_size = options.listing_page_size
    if explicit_limit is not None:
        try:
            page_size = max(1, min(int(explicit_limit), LISTING_PAGE_SIZE))
        except (TypeError, ValueError):
            page_size = options.listing_page_size

    if options.listing_limit is None:
        return page_size

    remaining = max(options.listing_limit - fetched_count, 0)
    if remaining == 0:
        return 0

    return max(1, min(page_size, remaining))


def _fetch_listing(
    session: requests.Session,
    target: ListingTarget,
    options: ScrapeOptions,
) -> tuple[Any, list[Any], int]:
    aggregated: Any | None = None
    combined_children: list[Any] = []
    cursor = target.params.get("after")
    fetched = 0
    pages = 0

    while True:
        if options.listing_limit is not None and fetched >= options.listing_limit:
            break

        params = dict(target.params)
        params.setdefault("raw_json", 1)
        page_limit = _determine_page_limit(target=target, options=options, fetched_count=fetched)
        if page_limit is not None:
            if page_limit == 0:
                break
            params["limit"] = page_limit
        if cursor:
            params["after"] = cursor

        listing_json = fetch_json(session, target.url, params=params)
        pages += 1

        if aggregated is None:
            aggregated = copy.deepcopy(listing_json)

        if not isinstance(listing_json, dict):
            break

        data = listing_json.get("data", {})
        children = data.get("children", []) or []
        combined_children.extend(children)
        fetched += len(children)

        cursor = data.get("after")
        if not cursor:
            break

    if aggregated is None or not isinstance(aggregated, dict):
        aggregated = {"data": {}}

    aggregated_data = aggregated.setdefault("data", {})
    if options.listing_limit is not None and len(combined_children) > options.listing_limit:
        combined_children = combined_children[: options.listing_limit]
    aggregated_data["children"] = combined_children
    aggregated_data["after"] = cursor

    return aggregated, combined_children, pages

def build_session(user_agent: str, verify: bool) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.8",
            "Referer": "https://www.reddit.com/",
            "Connection": "keep-alive",
        }
    )
    session.verify = verify
    return session


def fetch_json(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
    retries: int = 3,
    backoff: float = 1.0,
) -> Any:
    last_exc: Exception | None = None
    attempt = 0
    wait_seconds = 60
    while attempt < retries:
        try:
            response = session.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as exc:
            attempt += 1
            last_exc = exc
            if attempt >= retries:
                break
            wait_time = backoff * (2 ** (attempt - 1))
            logger.warning(
                "Request error fetching %s (attempt %d/%d): %s; retrying in %.1f seconds",
                url,
                attempt,
                retries,
                exc,
                wait_time,
            )
            time.sleep(wait_time)
            continue

        if response.status_code == 429:
            last_exc = RuntimeError("HTTP 429 Too Many Requests")
            logger.warning(
                "Rate limited fetching %s; waiting %d seconds before retrying.",
                url,
                wait_seconds,
            )
            time.sleep(wait_seconds)
            wait_seconds += 60
            continue

        attempt += 1

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            if attempt >= retries:
                break
            wait_time = backoff * (2 ** (attempt - 1))
            logger.warning(
                "HTTP error fetching %s (attempt %d/%d): %s; retrying in %.1f seconds",
                url,
                attempt,
                retries,
                exc,
                wait_time,
            )
            time.sleep(wait_time)
            continue

        try:
            return response.json()
        except ValueError as exc:
            last_exc = exc
            if attempt >= retries:
                break
            wait_time = backoff * (2 ** (attempt - 1))
            logger.warning(
                "Failed to decode JSON from %s (attempt %d/%d): %s; retrying in %.1f seconds",
                url,
                attempt,
                retries,
                exc,
                wait_time,
            )
            time.sleep(wait_time)
            continue
    raise RuntimeError(f"Failed to fetch {url!r}: {last_exc}") from last_exc


def _build_link_info(child_data: dict[str, Any], *, rank: int) -> dict[str, Any] | None:
    permalink = child_data.get("permalink")
    if not permalink:
        return None
    permalink = str(permalink)
    if not permalink.startswith("/"):
        permalink = "/" + permalink
    if not permalink.endswith("/"):
        permalink = permalink + "/"

    json_url = f"{BASE_URL}{permalink}.json"
    post_url = f"{BASE_URL}{permalink}"
    content_url = child_data.get("url_overridden_by_dest") or child_data.get("url") or post_url

    link_info: dict[str, Any] = {
        "rank": rank,
        "id": child_data.get("id"),
        "title": child_data.get("title"),
        "created_utc": child_data.get("created_utc"),
        "permalink": permalink,
        "url": json_url,
        "post_url": post_url,
        "content_url": content_url,
        "author": child_data.get("author"),
        "author_fullname": child_data.get("author_fullname"),
        "subreddit": child_data.get("subreddit"),
        "score": child_data.get("score"),
        "upvote_ratio": child_data.get("upvote_ratio"),
        "num_comments": child_data.get("num_comments"),
        "selftext": child_data.get("selftext"),
        "link_flair_text": child_data.get("link_flair_text"),
        "over_18": child_data.get("over_18"),
    }
    return link_info


def extract_links(listing_json: dict) -> List[dict[str, Any]]:
    children = listing_json.get("data", {}).get("children", [])
    links: List[dict[str, Any]] = []
    for rank, child in enumerate(children, start=1):
        child_data = child.get("data", {})
        link_info = _build_link_info(child_data, rank=rank)
        if not link_info:
            continue
        links.append(link_info)
    return links


def _extract_post_data(post_json: Any) -> dict[str, Any] | None:
    if isinstance(post_json, list) and post_json:
        first = post_json[0]
        if isinstance(first, dict):
            children = first.get("data", {}).get("children", []) or []
            if children:
                maybe_post = children[0].get("data")
                if isinstance(maybe_post, dict):
                    return maybe_post
    return None


def _find_existing_post_json(posts_dir: Path, link_info: dict[str, Any]) -> Path | None:
    if not posts_dir.exists():
        return None
    post_id = link_info.get("id")
    if not post_id:
        return None
    matches = sorted(posts_dir.glob(f"*{post_id}*.json"))
    if matches:
        return matches[0]
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "429" in message or "too many requests" in message or "rate limit" in message:
        return True
    if isinstance(exc, requests.HTTPError):
        try:
            status = exc.response.status_code if exc.response is not None else None
        except Exception:  # pragma: no cover - defensive
            status = None
        return status == 429
    return False


def _clean_media_url(url: Any) -> str | None:
    if not url:
        return None
    candidate = unescape(str(url)).strip()
    if not candidate:
        return None
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        return None
    return candidate


def _normalize_media_candidate(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path or ""
    lower_path = path.lower()

    if "poster.jpg" in lower_path or lower_path.endswith("thumbnail.jpg"):
        return None
    if host.endswith("redgifs.com") and lower_path.endswith(".jpg"):
        return None

    if lower_path.endswith(".gifv"):
        stem = Path(path).stem
        if not stem:
            return None
        if host == "imgur.com":
            return f"https://i.imgur.com/{stem}.mp4"
        if host.endswith("i.imgur.com"):
            return url[:-5] + ".mp4"
        if host.endswith("redditmedia.com"):
            return url[:-5] + ".mp4"
        return url[:-5] + ".mp4"

    return url


def _extension_priority(ext: str) -> int:
    ext = ext.lower()
    if ext in VIDEO_EXTENSIONS:
        return 3
    if ext in ANIMATED_IMAGE_EXTENSIONS:
        return 2
    if ext in STATIC_IMAGE_EXTENSIONS:
        return 1
    return 0


def _media_signature(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    posix_path = PurePosixPath(parsed.path or "")
    base = str(posix_path.with_suffix(""))
    base = base.lower().strip("/")
    if not base:
        base = parsed.path.lower().strip("/") or host
    return f"{host}::{base}"


def _classify_media_url(url: str) -> int:
    ext = _infer_extension_from_url(url)
    priority = _extension_priority(ext)
    if priority:
        return priority
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.endswith("v.redd.it"):
        return 3
    return 1


def _derive_reddit_audio_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not host.endswith("v.redd.it"):
        return None
    path = parsed.path or ""
    if "DASH_audio" in path:
        return None
    match = re.search(r"/DASH_[^/]+\.mp4$", path)
    if not match:
        return None
    audio_path = re.sub(r"/DASH_[^/]+\.mp4$", "/DASH_audio.mp4", path)
    return urlunparse(parsed._replace(path=audio_path))


def normalize_media_filter_tokens(filters: Iterable[str] | None) -> set[str] | None:
    if not filters:
        return None
    normalized: set[str] = set()
    for raw in filters:
        token = str(raw).strip().lower()
        if not token:
            continue
        if token in MEDIA_FILTER_CATEGORIES:
            normalized.add(token)
            continue
        candidate = token if token.startswith(".") else f".{token}"
        if candidate in MEDIA_EXTENSION_WHITELIST:
            normalized.add(candidate)
            continue
        raise ValueError(f"Unsupported media filter token: {raw}")
    return normalized or None


def _load_media_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse media manifest %s: %s", path, exc)
        return {}


def _persist_media_manifest(path: Path, manifest: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _should_download_media(url: str, allowed_filters: set[str] | None) -> bool:
    if not allowed_filters:
        return True

    categories: set[str] = set()
    ext = _infer_extension_from_url(url)
    parsed = urlparse(url)
    path_lower = (parsed.path or "").lower()
    if ext:
        if ext in VIDEO_EXTENSIONS:
            categories.add("video")
        elif ext in ANIMATED_IMAGE_EXTENSIONS:
            categories.add("animated")
        elif ext in STATIC_IMAGE_EXTENSIONS:
            categories.add("image")
    else:
        classification = _classify_media_url(url)
        if classification == 3:
            categories.add("video")
            ext = ".mp4"
        elif classification == 2:
            categories.add("animated")
            ext = ".gif"
        else:
            categories.add("image")
            ext = ".jpg"

    if "dash_audio" in path_lower:
        categories.add("audio")
        categories.discard("video")

    if ext and ext in allowed_filters:
        return True
    for category in categories:
        if category in allowed_filters:
            return True
    return False


def _infer_extension_from_url(url: str) -> str:
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.lower()
    if ext in MEDIA_EXTENSION_WHITELIST:
        if ext == ".gifv":
            return ".mp4"
        return ext
    return ""


def _infer_extension_from_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    content_type = content_type.split(";", 1)[0].lower()
    return CONTENT_TYPE_EXTENSION_MAP.get(content_type, "")


def _is_probable_media_url(url: str) -> bool:
    ext = _infer_extension_from_url(url)
    if ext:
        return True
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    # Known media hosts without obvious extensions
    if host.endswith("v.redd.it") or host.endswith("i.redd.it"):
        return True
    if host.endswith("imgur.com") and parsed.path:
        return True
    return False


def _collect_media_urls(
    link_info: dict[str, Any],
    child_data: dict[str, Any] | None,
    post_data: dict[str, Any] | None,
) -> list[str]:
    candidates: dict[str, tuple[int, int, str]] = {}
    order_counter = 0

    def add(raw_url: Any) -> None:
        nonlocal order_counter
        cleaned = _clean_media_url(raw_url)
        if not cleaned:
            return
        normalized = _normalize_media_candidate(cleaned)
        if not normalized:
            return
        if not _is_probable_media_url(normalized):
            return
        priority = _classify_media_url(normalized)
        signature = _media_signature(normalized)
        existing = candidates.get(signature)
        if existing is None:
            candidates[signature] = (priority, order_counter, normalized)
        else:
            current_priority, existing_order, existing_url = existing
            if priority > current_priority:
                candidates[signature] = (priority, existing_order, normalized)
        order_counter += 1

    def add_from_preview(preview: dict[str, Any] | None) -> None:
        if not isinstance(preview, dict):
            return
        for image in preview.get("images", []) or []:
            variants = image.get("variants", {}) or {}
            mp4_variant = variants.get("mp4")
            if isinstance(mp4_variant, dict):
                add(mp4_variant.get("source", {}).get("url"))
                continue
            gif_variant = variants.get("gif")
            if isinstance(gif_variant, dict):
                add(gif_variant.get("source", {}).get("url"))
                continue
            add(image.get("source", {}).get("url"))

    def add_from_media_metadata(metadata: dict[str, Any] | None, gallery_order: list[Any] | None) -> None:
        if not isinstance(metadata, dict):
            return
        ids: Iterable[str]
        if gallery_order:
            ids = [str(item.get("media_id")) for item in gallery_order if isinstance(item, dict)]
        else:
            ids = metadata.keys()
        for media_id in ids:
            media_entry = metadata.get(media_id)
            if not isinstance(media_entry, dict):
                continue
            source = media_entry.get("s") or {}
            add(source.get("mp4"))
            add(source.get("gif"))
            add(source.get("u"))
            gif_direct = media_entry.get("gif")
            if isinstance(gif_direct, str):
                add(gif_direct)

    def process_data(data: dict[str, Any] | None) -> None:
        if not isinstance(data, dict):
            return
        add(data.get("url_overridden_by_dest"))
        add(data.get("url"))
        secure_media = data.get("secure_media") or data.get("media") or {}
        if isinstance(secure_media, dict):
            reddit_video = secure_media.get("reddit_video")
            if isinstance(reddit_video, dict):
                fallback_url = reddit_video.get("fallback_url")
                add(fallback_url)
                audio_url = _derive_reddit_audio_url(fallback_url) if fallback_url else None
                if audio_url:
                    add(audio_url)
        add_from_preview(data.get("preview"))
        gallery_data = data.get("gallery_data", {}).get("items") if isinstance(data.get("gallery_data"), dict) else None
        add_from_media_metadata(data.get("media_metadata"), gallery_data)

    process_data(child_data)
    process_data(post_data)
    add(link_info.get("content_url"))

    ordered = sorted(candidates.values(), key=lambda item: item[1])
    return [entry[2] for entry in ordered]


def _download_single_media(
    session: requests.Session,
    url: str,
    dest_dir: Path,
    base_name: str,
    *,
    resume: bool,
    ext_hint: str,
) -> Path | None:
    try:
        with closing(session.get(url, stream=True, timeout=60)) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type") if hasattr(response, "headers") else None
            content_ext = _infer_extension_from_content_type(content_type)
            ext = ext_hint or ""
            if content_ext:
                if not ext:
                    ext = content_ext
                else:
                    if _extension_priority(content_ext) > _extension_priority(ext):
                        ext = content_ext
            if not ext:
                ext = ".bin"
            dest_path = dest_dir / f"{base_name}{ext}"
            if resume and dest_path.exists():
                logger.info("Media already exists, skipping %s", dest_path)
                return dest_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with dest_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            return dest_path
    except requests.exceptions.RequestException as exc:  # pragma: no cover - network errors
        logger.warning("Failed to download media %s: %s", url, exc)
    except Exception as exc:  # pragma: no cover - IO errors
        logger.warning("Failed to write media %s: %s", url, exc)
    return None


def _download_media_items(
    session: requests.Session,
    urls: list[str],
    *,
    media_dir: Path,
    base_name: str,
    downloaded_urls: set[str],
    resume: bool,
    manifest: dict[str, str] | None = None,
    manifest_path: Path | None = None,
    allowed_filters: set[str] | None = None,
) -> int:
    if not urls:
        return 0
    if manifest is None:
        manifest = {}
    saved = 0
    updated_manifest = False
    total = len(urls)
    for index, media_url in enumerate(urls, start=1):
        if media_url in downloaded_urls or media_url in manifest:
            downloaded_urls.add(media_url)
            continue
        if allowed_filters and not _should_download_media(media_url, allowed_filters):
            continue
        downloaded_urls.add(media_url)
        ext_hint = _infer_extension_from_url(media_url)
        suffix = f"_media{index:02d}" if total > 1 else "_media"
        candidate_name = shorten_component(f"{base_name}{suffix}", 140)
        result = _download_single_media(
            session,
            media_url,
            media_dir,
            candidate_name,
            resume=resume,
            ext_hint=ext_hint,
        )
        if result is not None:
            saved += 1
            try:
                relative_path = str(result.relative_to(media_dir))
            except ValueError:
                relative_path = result.name
            manifest[media_url] = relative_path
            updated_manifest = True
    if updated_manifest and manifest_path is not None:
        _persist_media_manifest(manifest_path, manifest)
    return saved

def derive_filename(link_info: dict[str, Any], post_data: dict[str, Any] | None) -> str:
    rank = link_info.get("rank")
    post_id = link_info.get("id") or "post"
    slug_source = link_info.get("title") or link_info.get("permalink") or "post"
    created_ts = link_info.get("created_utc")

    if post_data:
        post_id = post_data.get("id") or post_id
        created_ts = post_data.get("created_utc") or created_ts
        slug_source = post_data.get("title") or slug_source

    date_fragment = ""
    if created_ts:
        try:
            date_fragment = datetime.utcfromtimestamp(float(created_ts)).strftime("%Y%m%d")
        except (ValueError, TypeError, OverflowError):
            date_fragment = ""

    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(post_id)) or "post"
    safe_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", slug_source) or "post"
    safe_id = shorten_component(safe_id, 40)
    safe_slug = shorten_component(safe_slug, 100)

    parts: List[str] = []
    if rank is not None:
        try:
            parts.append(f"{int(rank):03d}")
        except (ValueError, TypeError):
            pass
    if date_fragment:
        parts.append(date_fragment)
    parts.append(safe_id)
    parts.append(safe_slug)

    return "_".join(parts) + ".json"


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(rows: List[dict[str, Any]], fieldnames: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_timestamp(epoch: Any) -> str:
    try:
        return datetime.utcfromtimestamp(float(epoch)).isoformat()
    except (TypeError, ValueError, OverflowError):
        return ""


def shorten_component(text: str, max_length: int = 80) -> str:
    if len(text) <= max_length:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    prefix_length = max(1, max_length - 9)
    return f"{text[:prefix_length]}_{digest}"


def flatten_post_record(
    subreddit: str, link_info: dict[str, Any], post_data: dict[str, Any] | None
) -> dict[str, Any]:
    post_data = post_data or {}
    permalink = post_data.get("permalink") or link_info.get("permalink") or ""
    if permalink and not permalink.startswith("http"):
        permalink = f"{BASE_URL}{permalink}"

    created_utc = post_data.get("created_utc") or link_info.get("created_utc")

    return {
        "rank": link_info.get("rank"),
        "post_id": post_data.get("id") or link_info.get("id"),
        "title": post_data.get("title") or link_info.get("title"),
        "author": post_data.get("author")
        or post_data.get("author_fullname")
        or link_info.get("author")
        or link_info.get("author_fullname"),
        "subreddit": post_data.get("subreddit") or subreddit,
        "created_utc": created_utc,
        "created_iso": format_timestamp(created_utc),
        "score": post_data.get("score") if post_data.get("score") is not None else link_info.get("score"),
        "upvote_ratio": post_data.get("upvote_ratio")
        if post_data.get("upvote_ratio") is not None
        else link_info.get("upvote_ratio"),
        "num_comments": post_data.get("num_comments")
        if post_data.get("num_comments") is not None
        else link_info.get("num_comments"),
        "permalink": permalink,
        "url": post_data.get("url_overridden_by_dest")
        or post_data.get("url")
        or link_info.get("content_url")
        or link_info.get("post_url")
        or link_info.get("url"),
        "selftext": post_data.get("selftext")
        if post_data.get("selftext") is not None
        else link_info.get("selftext"),
        "link_flair_text": post_data.get("link_flair_text")
        if post_data.get("link_flair_text") is not None
        else link_info.get("link_flair_text"),
        "over_18": post_data.get("over_18")
        if post_data.get("over_18") is not None
        else link_info.get("over_18"),
    }


def _flatten_comment_tree(
    nodes: Iterable[dict[str, Any]],
    *,
    post_context: dict[str, Any],
    depth: int,
    records: List[dict[str, Any]],
) -> None:
    for node in nodes:
        if not isinstance(node, dict):
            continue
        kind = node.get("kind")
        data = node.get("data", {})
        if kind != "t1":
            continue

        created_utc = data.get("created_utc")
        comment_permalink = data.get("permalink")
        if comment_permalink and not comment_permalink.startswith("http"):
            comment_permalink = f"{BASE_URL}{comment_permalink}"
        record = {
            "post_id": post_context.get("post_id"),
            "post_rank": post_context.get("rank"),
            "post_title": post_context.get("title"),
            "subreddit": post_context.get("subreddit"),
            "comment_id": data.get("id"),
            "parent_id": data.get("parent_id"),
            "author": data.get("author"),
            "created_utc": created_utc,
            "created_iso": format_timestamp(created_utc),
            "score": data.get("score"),
            "depth": depth,
            "body": data.get("body"),
            "permalink": comment_permalink,
            "is_submitter": data.get("is_submitter"),
        }
        records.append(record)

        replies = data.get("replies")
        if isinstance(replies, dict):
            children = replies.get("data", {}).get("children", [])
            _flatten_comment_tree(
                children,
                post_context=post_context,
                depth=depth + 1,
                records=records,
            )


def flatten_comments(post_json: Any, post_context: dict[str, Any]) -> List[dict[str, Any]]:
    if not isinstance(post_json, list) or len(post_json) < 2:
        return []
    comment_listing = post_json[1]
    if not isinstance(comment_listing, dict):
        return []
    children = comment_listing.get("data", {}).get("children", [])
    records: List[dict[str, Any]] = []
    _flatten_comment_tree(children, post_context=post_context, depth=0, records=records)
    return records


def rebuild_csv_from_cache(target: ListingTarget, output_root: Path) -> None:
    """Recreate CSV summaries from previously saved JSON artifacts."""
    base_dir = target.output_dir(output_root)
    posts_dir = base_dir / "post_jsons"
    links_path = base_dir / "links.json"
    if not posts_dir.exists():
        raise FileNotFoundError(
            f"No cached post_jsons/ directory for {target.label} (expected {posts_dir})"
        )

    links: list[dict[str, Any]] = []
    if links_path.exists():
        try:
            links = json.loads(links_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse %s: %s. Continuing without link metadata.", links_path, exc)

    link_map: dict[str, dict[str, Any]] = {}
    for entry in links:
        if isinstance(entry, dict):
            post_id = str(entry.get("id") or "").strip()
            if post_id:
                link_map[post_id] = entry

    posts_records: List[dict[str, Any]] = []
    comments_records: List[dict[str, Any]] = []

    json_files = sorted(posts_dir.glob("*.json"))
    for idx, path in enumerate(json_files, start=1):
        try:
            post_json = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping %s: invalid JSON (%s)", path.name, exc)
            continue

        post_data: dict[str, Any] | None = None
        if isinstance(post_json, list) and post_json:
            first = post_json[0]
            if isinstance(first, dict):
                children = first.get("data", {}).get("children", [])
                if children:
                    maybe_post = children[0].get("data")
                    if isinstance(maybe_post, dict):
                        post_data = maybe_post
        if post_data is None:
            post_data = {}

        post_id = str(post_data.get("id") or "").strip()
        link_info = link_map.get(post_id, {})
        if not link_info:
            link_info = _build_link_info(post_data, rank=idx) or {
                "rank": idx,
                "id": post_data.get("id") or path.stem,
                "title": post_data.get("title"),
                "created_utc": post_data.get("created_utc"),
                "permalink": post_data.get("permalink"),
                "url": post_data.get("permalink"),
            }
        else:
            link_info = dict(link_info)
            link_info.setdefault("rank", idx)

        post_record = flatten_post_record(target.resolved_context(), link_info, post_data)
        posts_records.append(post_record)
        comments_records.extend(
            flatten_comments(
                post_json,
                {
                    "post_id": post_record.get("post_id"),
                    "rank": post_record.get("rank"),
                    "title": post_record.get("title"),
                    "subreddit": post_record.get("subreddit"),
                },
            )
        )

    if not posts_records:
        logger.warning("No posts reconstructed for %s", target.label)
        return

    def _rank_key(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    posts_records.sort(key=lambda r: (_rank_key(r.get("rank")), r.get("created_utc") or 0))

    posts_csv_path = base_dir / "posts.csv"
    comments_csv_path = base_dir / "comments.csv"
    write_csv(
        posts_records,
        [
            "rank",
            "post_id",
            "title",
            "author",
            "subreddit",
            "created_utc",
            "created_iso",
            "score",
            "upvote_ratio",
            "num_comments",
            "permalink",
            "url",
            "selftext",
            "link_flair_text",
            "over_18",
        ],
        posts_csv_path,
    )
    write_csv(
        comments_records,
        [
            "post_id",
            "post_rank",
            "post_title",
            "subreddit",
            "comment_id",
            "parent_id",
            "author",
            "created_utc",
            "created_iso",
            "score",
            "depth",
            "body",
            "permalink",
            "is_submitter",
        ],
        comments_csv_path,
    )
    logger.info(
        "Rebuilt CSV summaries for %s at %s and %s",
        target.label,
        posts_csv_path,
        comments_csv_path,
    )


def process_listing(
    target: ListingTarget,
    *,
    session: requests.Session,
    options: ScrapeOptions,
) -> None:
    logger.info("=== Processing %s ===", target.label)
    base_dir = target.output_dir(options.output_root)
    posts_path = base_dir / "posts.json"
    links_path = base_dir / "links.json"
    posts_dir = base_dir / "post_jsons"
    media_dir = base_dir / "media"
    media_manifest_path = base_dir / "media_manifest.json"
    media_manifest: dict[str, str] = {}
    downloaded_media_urls: set[str] = set()
    if options.download_media:
        if options.resume:
            media_manifest = _load_media_manifest(media_manifest_path)
        downloaded_media_urls = set(media_manifest.keys())

    listing_json, children, pages = _fetch_listing(session, target, options)
    if "json" in options.output_formats:
        save_json(listing_json, posts_path)
        logger.info(
            "Saved listing to %s (%d items across %d page(s))",
            posts_path,
            len(children),
            pages,
        )
    else:
        logger.info(
            "Fetched listing for %s (%d items, %d page(s))",
            target.label,
            len(children),
            pages,
        )

    child_lookup: dict[str, dict[str, Any]] = {}
    for child in children:
        if not isinstance(child, dict):
            continue
        data = child.get("data")
        if not isinstance(data, dict):
            continue
        post_id = str(data.get("id") or "").strip()
        if post_id:
            child_lookup[post_id] = data

    links = extract_links(listing_json)
    if options.listing_limit is not None:
        links = links[: options.listing_limit]
    if "json" in options.output_formats:
        save_json(links, links_path)
        logger.info("Saved %d links to %s", len(links), links_path)
    else:
        logger.info("Extracted %d post permalinks", len(links))

    total_links = len(links)
    posts_records: List[dict[str, Any]] = []
    comments_records: List[dict[str, Any]] = []
    fetched_count = 0
    reused_count = 0
    skipped_count = 0
    rate_limit_events = 0
    media_saved = 0

    for link_info in links:
        if not options.fetch_comments:
            if "csv" in options.output_formats:
                post_record = flatten_post_record(target.resolved_context(), link_info, None)
                posts_records.append(post_record)
            if options.download_media:
                post_id = str(link_info.get("id") or "")
                child_data = child_lookup.get(post_id, {})
                media_urls = _collect_media_urls(link_info, child_data, None)
                media_saved += _download_media_items(
                    session,
                    media_urls,
                    media_dir=media_dir,
                    base_name=shorten_component(f"{link_info.get('id') or 'post'}", 120),
                    downloaded_urls=downloaded_media_urls,
                    resume=options.resume,
                    manifest=media_manifest,
                    manifest_path=media_manifest_path,
                    allowed_filters=options.media_filters,
                )
            continue

        post_url = link_info.get("url")
        if not post_url:
            continue

        rank_label = link_info.get("rank")
        progress = f"{rank_label}/{total_links}" if rank_label else f"?/{total_links}"

        cached_path: Path | None = None
        post_json: Any | None = None
        post_data: dict[str, Any] | None = None
        reused = False

        if options.resume:
            cached_path = _find_existing_post_json(posts_dir, link_info)
            if cached_path:
                try:
                    post_json = json.loads(cached_path.read_text(encoding="utf-8"))
                    post_data = _extract_post_data(post_json) or {}
                    reused = True
                    if "json" not in options.output_formats:
                        logger.info("[%s] Using cached data for %s", progress, post_url)
                except Exception as cache_exc:  # noqa: BLE001 - treat as cache miss
                    logger.warning(
                        "Failed to reuse cached post %s: %s",
                        cached_path,
                        cache_exc,
                    )
                    post_json = None
                    post_data = None
                    cached_path = None

        attempted_fetch = False
        fetched_successfully = False

        if post_json is None:
            params = {"raw_json": 1, "limit": options.comment_limit}
            wait_seconds = 60
            while True:
                attempted_fetch = True
                try:
                    post_json = fetch_json(session, post_url, params=params)
                    fetched_successfully = True
                    break
                except Exception as exc:  # noqa: BLE001 - handle rate limit and continue
                    if _is_rate_limit_error(exc):
                        rate_limit_events += 1
                        logger.warning(
                            "Rate limited fetching %s; waiting %d seconds before retrying.",
                            post_url,
                            wait_seconds,
                        )
                        time.sleep(wait_seconds)
                        wait_seconds += 60
                        continue
                    logger.warning("Skipping post at %s due to error: %s", post_url, exc)
                    post_json = None
                    break

        if post_json is None:
            if attempted_fetch:
                time.sleep(options.delay)
                skipped_count += 1
            continue

        post_data = post_data or _extract_post_data(post_json) or {}

        if cached_path:
            target_path = cached_path
            base_name = Path(cached_path).stem
        else:
            derived_filename = derive_filename(link_info, post_data)
            target_path = posts_dir / derived_filename
            base_name = Path(derived_filename).stem

        if "json" in options.output_formats:
            if reused and cached_path:
                logger.info("[%s] Cached post already stored at %s", progress, target_path)
            else:
                logger.info("[%s] Fetching %s -> %s", progress, post_url, target_path)
                save_json(post_json, target_path)
        else:
            if not reused:
                logger.info("[%s] Fetching %s", progress, post_url)

        if "csv" in options.output_formats:
            post_record = flatten_post_record(target.resolved_context(), link_info, post_data)
            posts_records.append(post_record)
            comments_records.extend(
                flatten_comments(
                    post_json,
                    {
                        "post_id": post_record.get("post_id"),
                        "rank": post_record.get("rank"),
                        "title": post_record.get("title"),
                        "subreddit": post_record.get("subreddit"),
                    },
                )
            )

        if reused:
            reused_count += 1

        if attempted_fetch and fetched_successfully:
            fetched_count += 1
            time.sleep(options.delay)

        if options.download_media:
            post_id = str(link_info.get("id") or "")
            child_data = child_lookup.get(post_id, {})
            media_urls = _collect_media_urls(link_info, child_data, post_data)
            media_saved += _download_media_items(
                session,
                media_urls,
                media_dir=media_dir,
                base_name=base_name,
                downloaded_urls=downloaded_media_urls,
                resume=options.resume,
                manifest=media_manifest,
                manifest_path=media_manifest_path,
                allowed_filters=options.media_filters,
            )

    if "csv" in options.output_formats:
        posts_csv_path = base_dir / "posts.csv"
        comments_csv_path = base_dir / "comments.csv"
        write_csv(
            posts_records,
            [
                "rank",
                "post_id",
                "title",
                "author",
                "subreddit",
                "created_utc",
                "created_iso",
                "score",
                "upvote_ratio",
                "num_comments",
                "permalink",
                "url",
                "selftext",
                "link_flair_text",
                "over_18",
            ],
            posts_csv_path,
        )
        if options.fetch_comments:
            write_csv(
                comments_records,
                [
                    "post_id",
                    "post_rank",
                    "post_title",
                    "subreddit",
                    "comment_id",
                    "parent_id",
                    "author",
                    "created_utc",
                    "created_iso",
                    "score",
                    "depth",
                    "body",
                    "permalink",
                    "is_submitter",
                ],
                comments_csv_path,
            )
            logger.info(
                "Wrote CSV summaries to %s and %s",
                posts_csv_path,
                comments_csv_path,
            )
        else:
            logger.info("Wrote CSV summary to %s", posts_csv_path)

    logger.info(
        "Completed %s: total=%d fetched=%d reused=%d skipped=%d rate-limit-waits=%d comments=%d media=%d",
        target.label,
        total_links,
        fetched_count,
        reused_count,
        skipped_count,
        rate_limit_events,
        len(comments_records),
        media_saved,
    )


def process_post(
    target: PostTarget,
    *,
    session: requests.Session,
    options: ScrapeOptions,
) -> None:
    logger.info("=== Processing %s ===", target.label)
    base_dir = target.output_dir(options.output_root)
    posts_dir = base_dir / "post_jsons"
    links_path = base_dir / "links.json"
    media_dir = base_dir / "media"
    media_manifest_path = base_dir / "media_manifest.json"
    media_manifest: dict[str, str] = {}
    downloaded_media_urls: set[str] = set()
    if options.download_media and options.resume:
        media_manifest = _load_media_manifest(media_manifest_path)
        downloaded_media_urls = set(media_manifest.keys())

    params = {"raw_json": 1, "limit": options.comment_limit}
    params.update(target.params)
    post_json = fetch_json(session, target.url, params=params)

    post_data: dict[str, Any] | None = None
    if isinstance(post_json, list) and post_json:
        first = post_json[0]
        if isinstance(first, dict):
            children = first.get("data", {}).get("children", [])
            if children:
                maybe_post = children[0].get("data")
                if isinstance(maybe_post, dict):
                    post_data = maybe_post
    if post_data is None:
        post_data = {}

    link_info = _build_link_info(post_data, rank=1) or {
        "rank": 1,
        "id": post_data.get("id"),
        "title": post_data.get("title"),
        "created_utc": post_data.get("created_utc"),
        "permalink": post_data.get("permalink"),
        "url": target.url,
        "post_url": target.url,
        "content_url": target.url,
    }
    links = [link_info]

    filename = derive_filename(link_info, post_data)
    base_name = Path(filename).stem

    if "json" in options.output_formats:
        target_path = posts_dir / filename
        logger.info("Saving post JSON to %s", target_path)
        save_json(post_json, target_path)
        save_json(links, links_path)
    else:
        logger.info("Fetched post JSON")

    post_record = flatten_post_record(target.context or link_info.get("subreddit") or "post", link_info, post_data)
    comments_records = flatten_comments(
        post_json,
        {
            "post_id": post_record.get("post_id"),
            "rank": post_record.get("rank"),
            "title": post_record.get("title"),
            "subreddit": post_record.get("subreddit"),
        },
    )

    if "csv" in options.output_formats:
        posts_csv_path = base_dir / "posts.csv"
        comments_csv_path = base_dir / "comments.csv"
        write_csv(
            [post_record],
            [
                "rank",
                "post_id",
                "title",
                "author",
                "subreddit",
                "created_utc",
                "created_iso",
                "score",
                "upvote_ratio",
                "num_comments",
                "permalink",
                "url",
                "selftext",
                "link_flair_text",
                "over_18",
            ],
            posts_csv_path,
        )
        if comments_records:
            write_csv(
                comments_records,
                [
                    "post_id",
                    "post_rank",
                    "post_title",
                    "subreddit",
                    "comment_id",
                    "parent_id",
                    "author",
                    "created_utc",
                    "created_iso",
                    "score",
                    "depth",
                    "body",
                    "permalink",
                    "is_submitter",
                ],
                comments_csv_path,
            )
        logger.info(
            "Wrote outputs to %s (comments=%d)",
            base_dir,
            len(comments_records),
        )

    media_saved = 0
    if options.download_media:
        media_urls = _collect_media_urls(link_info, post_data, post_data)
        media_saved = _download_media_items(
            session,
            media_urls,
            media_dir=media_dir,
            base_name=base_name,
            downloaded_urls=downloaded_media_urls,
            resume=options.resume,
            manifest=media_manifest,
            manifest_path=media_manifest_path,
            allowed_filters=options.media_filters,
        )
        logger.info("Downloaded %d media file(s) for %s", media_saved, target.label)

    logger.info(
        "Completed %s (comments=%d, media=%d)",
        target.label,
        len(comments_records),
        media_saved,
    )
__all__ = [
    "BASE_URL",
    "DEFAULT_USER_AGENT",
    "ListingTarget",
    "PostTarget",
    "ScrapeOptions",
    "build_session",
    "fetch_json",
    "extract_links",
    "derive_filename",
    "flatten_comments",
    "flatten_post_record",
    "format_timestamp",
    "process_listing",
    "process_post",
    "normalize_media_filter_tokens",
    "rebuild_csv_from_cache",
    "save_json",
    "shorten_component",
    "write_csv",
]
