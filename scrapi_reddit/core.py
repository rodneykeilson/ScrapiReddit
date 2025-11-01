"""Core scraping utilities for the Scrapi Reddit toolkit."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List

import requests

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
BASE_URL = "https://www.reddit.com"
MAX_LISTING_LIMIT = 100
MAX_COMMENT_LIMIT = 500
MIN_DELAY_SECONDS = 1.0


@dataclass(slots=True)
class ScrapeOptions:
    """Configuration shared by subreddit processing routines."""

    output_root: Path
    listing_limit: int
    comment_limit: int
    delay: float
    time_filter: str
    output_formats: set[str] = field(default_factory=lambda: {"json"})

    def __post_init__(self) -> None:
        self.listing_limit = max(1, min(self.listing_limit, MAX_LISTING_LIMIT))
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
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=30)
            if response.status_code == 429:
                last_exc = RuntimeError("HTTP 429 Too Many Requests")
                retry_after_header = response.headers.get("Retry-After")
                if retry_after_header:
                    try:
                        wait_time = float(retry_after_header)
                    except ValueError:
                        wait_time = backoff * (2 ** (attempt - 1))
                else:
                    wait_time = backoff * (2 ** (attempt - 1))
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            try:
                return response.json()
            except ValueError as exc:
                last_exc = exc
                if attempt == retries:
                    break
                wait_time = backoff * (2 ** (attempt - 1))
                time.sleep(wait_time)
                continue
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt == retries:
                break
            wait_time = backoff * (2 ** (attempt - 1))
            time.sleep(wait_time)
    raise RuntimeError(f"Failed to fetch {url!r}: {last_exc}") from last_exc


def extract_links(listing_json: dict) -> List[dict[str, Any]]:
    children = listing_json.get("data", {}).get("children", [])
    links: List[dict[str, Any]] = []
    for rank, child in enumerate(children, start=1):
        child_data = child.get("data", {})
        permalink = child_data.get("permalink")
        if not permalink:
            continue
        if not permalink.startswith("/"):
            permalink = "/" + permalink
        if not permalink.endswith("/"):
            permalink = permalink + "/"
        links.append(
            {
                "rank": rank,
                "id": child_data.get("id"),
                "title": child_data.get("title"),
                "created_utc": child_data.get("created_utc"),
                "permalink": permalink,
                "url": f"{BASE_URL}{permalink}.json",
            }
        )
    return links


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
        "author": post_data.get("author") or post_data.get("author_fullname"),
        "subreddit": post_data.get("subreddit") or subreddit,
        "created_utc": created_utc,
        "created_iso": format_timestamp(created_utc),
        "score": post_data.get("score"),
        "upvote_ratio": post_data.get("upvote_ratio"),
        "num_comments": post_data.get("num_comments"),
        "permalink": permalink,
        "url": post_data.get("url_overridden_by_dest")
        or post_data.get("url")
        or link_info.get("url"),
        "selftext": post_data.get("selftext"),
        "link_flair_text": post_data.get("link_flair_text"),
        "over_18": post_data.get("over_18"),
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
            print(f"[WARN] Failed to parse {links_path}: {exc}. Continuing without link metadata.")

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
            print(f"[WARN] Skipping {path.name}: invalid JSON ({exc})")
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
            link_info = {
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
        print(f"[WARN] No posts reconstructed for {target.label}")
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
    print(f"Rebuilt CSV summaries for {target.label} at {posts_csv_path} and {comments_csv_path}")
def process_listing(
    target: ListingTarget,
    *,
    session: requests.Session,
    options: ScrapeOptions,
) -> None:
    print(f"\n=== Processing {target.label} ===")
    base_dir = target.output_dir(options.output_root)
    posts_path = base_dir / "posts.json"
    links_path = base_dir / "links.json"
    posts_dir = base_dir / "post_jsons"

    listing_params = dict(target.params)
    if target.allow_limit and "limit" not in listing_params:
        listing_params["limit"] = options.listing_limit
    listing_params.setdefault("raw_json", 1)
    listing_json = fetch_json(session, target.url, params=listing_params)
    if "json" in options.output_formats:
        save_json(listing_json, posts_path)
        print(f"Saved listing to {posts_path}")
    else:
        print(f"Fetched listing for {target.label}")

    links = extract_links(listing_json)
    if "json" in options.output_formats:
        save_json(links, links_path)
        print(f"Saved {len(links)} links to {links_path}")
    else:
        print(f"Extracted {len(links)} post permalinks")

    total_links = len(links)
    posts_records: List[dict[str, Any]] = []
    comments_records: List[dict[str, Any]] = []

    for link_info in links:
        post_url = link_info.get("url")
        if not post_url:
            continue
        rank_label = link_info.get("rank")
        progress = f"{rank_label}/{total_links}" if rank_label else f"?/{total_links}"
        params = {"raw_json": 1, "limit": options.comment_limit}
        try:
            post_json = fetch_json(session, post_url, params=params)
        except Exception as exc:  # noqa: BLE001 - continue scraping other links
            print(f"[WARN] Skipping post at {post_url} due to error: {exc}", file=sys.stderr)
            time.sleep(options.delay)
            continue

        post_data = None
        if isinstance(post_json, list) and post_json:
            first = post_json[0]
            if isinstance(first, dict):
                children = first.get("data", {}).get("children", [])
                if children:
                    post_data = children[0].get("data")

        if "json" in options.output_formats:
            filename = derive_filename(link_info, post_data)
            target_path = posts_dir / filename
            print(f"[{progress}] Fetching {post_url} -> {target_path}")
            save_json(post_json, target_path)
        else:
            print(f"[{progress}] Fetching {post_url}")

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
        time.sleep(options.delay)

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
        print(f"Wrote CSV summaries to {posts_csv_path} and {comments_csv_path}")


__all__ = [
    "BASE_URL",
    "DEFAULT_USER_AGENT",
    "ListingTarget",
    "ScrapeOptions",
    "build_session",
    "fetch_json",
    "extract_links",
    "derive_filename",
    "flatten_comments",
    "flatten_post_record",
    "format_timestamp",
    "process_listing",
    "rebuild_csv_from_cache",
    "save_json",
    "shorten_component",
    "write_csv",
]
