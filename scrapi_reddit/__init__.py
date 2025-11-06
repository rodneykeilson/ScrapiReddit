"""Public package surface for Scrapi Reddit."""
from .core import (
    BASE_URL,
    DEFAULT_USER_AGENT,
    build_search_target,
    ListingTarget,
    PostTarget,
    ScrapeOptions,
    build_session,
    extract_links,
    fetch_json,
    flatten_comments,
    flatten_post_record,
    format_timestamp,
    process_listing,
    process_post,
    rebuild_csv_from_cache,
    shorten_component,
)

__version__ = "0.1.0"

__all__ = [
    "BASE_URL",
    "DEFAULT_USER_AGENT",
    "ListingTarget",
    "PostTarget",
    "ScrapeOptions",
    "build_search_target",
    "build_session",
    "extract_links",
    "fetch_json",
    "flatten_comments",
    "flatten_post_record",
    "format_timestamp",
    "process_listing",
    "process_post",
    "rebuild_csv_from_cache",
    "shorten_component",
    "__version__",
]
