"""Command line entry point for the Scrapi Reddit scraper."""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Sequence
from urllib.parse import parse_qsl, urlsplit

from .core import (
    BASE_URL,
    DEFAULT_USER_AGENT,
    ListingTarget,
    PostTarget,
    ScrapeOptions,
    build_session,
    process_listing,
    process_post,
    rebuild_csv_from_cache,
    shorten_component,
    normalize_media_filter_tokens,
)


def _default_output_root() -> Path:
    env_override = os.environ.get("SCRAPI_REDDIT_OUTPUT_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return Path.cwd() / "scrapi_reddit_data"


def _resolve_subreddits(args_subreddits: Sequence[str], prompt: bool) -> List[str]:
    if args_subreddits:
        return [name.strip() for name in args_subreddits if name.strip()]
    if prompt:
        raw = input("Enter subreddit names (comma-separated): ").strip()
        return [name.strip() for name in raw.split(",") if name.strip()]
    return []


def _parse_csv(value: str, *, default: str | None = None) -> List[str]:
    raw = value if value is not None else default
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _validate_choices(values: List[str], allowed: set[str], option_name: str) -> None:
    invalid = [v for v in values if v not in allowed]
    if invalid:
        allowed_list = ", ".join(sorted(allowed))
        raise SystemExit(f"Invalid value(s) for {option_name}: {invalid}. Allowed: {allowed_list}")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    return slug or "item"


def _target_from_url(url: str) -> ListingTarget:
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        raise SystemExit(f"Listing URL must be absolute (including https://): {url}")
    base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path or '/'}"
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    slug_source = parsed.netloc + (parsed.path or "/")
    if parsed.query:
        slug_source += "?" + parsed.query
    slug = shorten_component(_slugify(slug_source), 80)
    return ListingTarget(
        label=f"custom {parsed.netloc}{parsed.path}",
        output_segments=("custom", slug),
        url=base_url,
        params=params,
        context="custom",
    )


def _post_target_from_url(url: str) -> PostTarget:
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        raise SystemExit(f"Post URL must be absolute (including https://): {url}")

    path = parsed.path or "/"
    segments = [segment for segment in path.split("/") if segment]

    context = parsed.netloc
    if segments and segments[0].lower() == "r" and len(segments) >= 2:
        context = f"r/{segments[1]}"
    elif segments and segments[0].lower() in {"u", "user"} and len(segments) >= 2:
        context = f"u/{segments[1]}"

    post_id: str | None = None
    try:
        comments_index = next(i for i, segment in enumerate(segments) if segment.lower() == "comments")
    except StopIteration:
        comments_index = -1
    if comments_index != -1 and len(segments) > comments_index + 1:
        post_id = segments[comments_index + 1]

    slug_source_parts: List[str] = []
    if post_id:
        slug_source_parts.append(post_id)
    if segments:
        slug_source_parts.append(segments[-1])
    if not slug_source_parts:
        slug_source_parts.append(parsed.netloc)
    slug_source = "-".join(slug_source_parts)
    slug = shorten_component(_slugify(slug_source), 80)

    json_path = path
    if not json_path.endswith("/"):
        json_path += "/"
    if not json_path.endswith(".json"):
        json_path += ".json"

    params = dict(parse_qsl(parsed.query, keep_blank_values=True))

    label = f"post {context}"
    context_slug = shorten_component(_slugify(context or parsed.netloc), 40)

    url_json = f"{parsed.scheme}://{parsed.netloc}{json_path}"

    return PostTarget(
        label=label,
        output_segments=("posts", context_slug, slug),
        url=url_json,
        params=params,
        context=context,
    )


def _build_targets(
    subreddits: List[str],
    *,
    subreddit_sorts: List[str],
    subreddit_top_times: List[str],
    include_frontpage: bool,
    include_r_all: bool,
    include_popular: bool,
    popular_sorts: List[str],
    popular_top_times: List[str],
    popular_geo: List[str],
    popular_geo_sorts: List[str],
    users: List[str],
    user_sections: List[str],
    user_sorts: List[str],
    listing_urls: List[str],
) -> List[ListingTarget]:
    targets: List[ListingTarget] = []

    allowed_subreddit_sorts = {"top", "best", "hot", "new", "rising"}
    _validate_choices(subreddit_sorts, allowed_subreddit_sorts, "--subreddit-sorts")
    allowed_time_filters = {"hour", "day", "week", "month", "year", "all"}
    if "top" in subreddit_sorts:
        _validate_choices(subreddit_top_times, allowed_time_filters, "--subreddit-top-times")

    for subreddit in subreddits:
        slug_name = _slugify(subreddit)
        for sort in subreddit_sorts:
            if sort == "top":
                for timeframe in subreddit_top_times:
                    targets.append(
                        ListingTarget(
                            label=f"r/{subreddit} top ({timeframe})",
                            output_segments=(
                                "subreddits",
                                slug_name,
                                f"top_{_slugify(timeframe)}",
                            ),
                            url=f"{BASE_URL}/r/{subreddit}/top/.json",
                            params={"t": timeframe},
                            context=subreddit,
                        )
                    )
            else:
                targets.append(
                    ListingTarget(
                        label=f"r/{subreddit} {sort}",
                        output_segments=("subreddits", slug_name, sort),
                        url=f"{BASE_URL}/r/{subreddit}/{sort}/.json",
                        context=subreddit,
                    )
                )

    if include_frontpage:
        targets.append(
            ListingTarget(
                label="reddit.com front page",
                output_segments=("frontpage", "default"),
                url=f"{BASE_URL}/.json",
                context="frontpage",
            )
        )

    if include_r_all:
        targets.append(
            ListingTarget(
                label="r/all",
                output_segments=("all", "default"),
                url=f"{BASE_URL}/r/all/.json",
                context="all",
            )
        )

    allowed_popular_sorts = {"best", "hot", "new", "top", "rising"}
    _validate_choices(popular_sorts, allowed_popular_sorts, "--popular-sorts")
    if include_popular:
        for sort in popular_sorts:
            if sort == "top":
                _validate_choices(popular_top_times, allowed_time_filters, "--popular-top-times")
                for timeframe in popular_top_times:
                    targets.append(
                        ListingTarget(
                            label=f"r/popular top ({timeframe})",
                            output_segments=("popular", "top", _slugify(timeframe)),
                            url=f"{BASE_URL}/r/popular/top/.json",
                            params={"t": timeframe},
                            context="popular",
                        )
                    )
            else:
                targets.append(
                    ListingTarget(
                        label=f"r/popular {sort}",
                        output_segments=("popular", sort),
                        url=f"{BASE_URL}/r/popular/{sort}/.json",
                        context="popular",
                    )
                )

    if popular_geo:
        allowed_popular_geo_sorts = {"best", "hot", "new", "top", "rising"}
        _validate_choices(popular_geo_sorts, allowed_popular_geo_sorts, "--popular-geo-sorts")
        for geo in popular_geo:
            code = geo.lower()
            geo_slug = _slugify(code)
            for sort in popular_geo_sorts:
                if sort == "top":
                    _validate_choices(popular_top_times, allowed_time_filters, "--popular-top-times")
                    for timeframe in popular_top_times:
                        targets.append(
                            ListingTarget(
                                label=f"r/popular top (geo={code}, {timeframe})",
                                output_segments=(
                                    "popular",
                                    "geo",
                                    geo_slug,
                                    "top",
                                    _slugify(timeframe),
                                ),
                                url=f"{BASE_URL}/r/popular/top/.json",
                                params={"geo_filter": code, "t": timeframe},
                                context="popular",
                            )
                        )
                else:
                    targets.append(
                        ListingTarget(
                            label=f"r/popular {sort} (geo={code})",
                            output_segments=("popular", "geo", geo_slug, sort),
                            url=f"{BASE_URL}/r/popular/{sort}/.json",
                            params={"geo_filter": code},
                            context="popular",
                        )
                    )

    allowed_user_sections = {"overview", "submitted", "comments"}
    _validate_choices(user_sections, allowed_user_sections, "--user-sections")
    allowed_user_sorts = {"new", "hot", "top", "best"}
    _validate_choices(user_sorts, allowed_user_sorts, "--user-sorts")
    for user in users:
        slug_user = _slugify(user)
        for section in user_sections:
            if section == "overview":
                path = f"/user/{user}/.json"
            else:
                path = f"/user/{user}/{section}/.json"
            for sort in user_sorts:
                targets.append(
                    ListingTarget(
                        label=f"u/{user} {section} ({sort})",
                        output_segments=("users", slug_user, section, sort),
                        url=f"{BASE_URL}{path}",
                        params={"sort": sort},
                        context=f"u/{user}",
                    )
                )

    for url in listing_urls:
        targets.append(_target_from_url(url))

    return targets


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Reddit listings for one or more subreddits. Respect Reddit rate limits; "
            "the tool enforces a minimum one-second delay between post requests."
        )
    )
    parser.add_argument(
        "subreddits",
        nargs="*",
        help="Subreddit names (without the r/ prefix).",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Prompt interactively for subreddit names when none are provided.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help=(
            "Maximum number of items to fetch per listing (total). Values above 100 trigger paginated "
            "fetches; use 0 for no explicit cap (default: 100)."
        ),
    )
    parser.add_argument(
        "--comment-limit",
        type=int,
        default=250,
        help="Comments per post request (default: 250, max: 500).",
    )
    parser.add_argument(
        "--fetch-comments",
        action="store_true",
        help="Fetch comments for listing posts (default: disabled to reduce API calls).",
    )
    parser.add_argument(
        "--download-media",
        action="store_true",
        help="Download linked images, GIFs, and videos for each scraped post.",
    )
    parser.add_argument(
        "--media-filter",
        default=None,
        help=(
            "Comma-separated list of media categories or extensions to download when --download-media is set. "
            "Supported categories: video, image, animated, audio; extensions: mp4, webm, gif, jpg, png, etc."
        ),
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help=(
            "Resume a previous run by reusing cached post_jsons files when present, skipping redundant fetches."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Delay in seconds between individual post requests (minimum enforced: 1.0).",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="Custom User-Agent header to send with requests.",
    )
    parser.add_argument(
        "--time-filter",
        choices=["hour", "day", "week", "month", "year", "all"],
        default="day",
        help="Which 'top' timeframe to use (default: day).",
    )
    parser.add_argument(
        "--subreddit-sorts",
        default="top",
        help=(
            "Comma-separated listing sorts for subreddit targets (choices: best, hot, new, rising, top)."
        ),
    )
    parser.add_argument(
        "--subreddit-top-times",
        default=None,
        help=(
            "Comma-separated time filters for subreddit 'top' listings (choices: hour, day, week, month, year, all)."
        ),
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (only if you trust the network).",
    )
    parser.add_argument(
        "--rebuild-from-json",
        action="store_true",
        help="Recreate CSV outputs from previously saved JSON files without new network calls.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "both"],
        default="json",
        help="Persist results as JSON files, CSV summaries, or both (default: json).",
    )
    parser.add_argument(
        "--frontpage",
        action="store_true",
        help="Include the reddit.com front page (.json) listing in the scrape run.",
    )
    parser.add_argument(
        "--include-r-all",
        action="store_true",
        help="Include the r/all listing in the scrape run.",
    )
    parser.add_argument(
        "--popular",
        action="store_true",
        help="Include r/popular listings using the sorts configured via --popular-sorts.",
    )
    parser.add_argument(
        "--popular-sorts",
        default="best,hot,new,top,rising",
        help=(
            "Comma-separated sorts for r/popular when --popular is provided (choices: best, hot, new, top, rising)."
        ),
    )
    parser.add_argument(
        "--popular-top-times",
        default=None,
        help=(
            "Comma-separated time filters for r/popular 'top' listings (choices: hour, day, week, month, year, all)."
        ),
    )
    parser.add_argument(
        "--popular-geo",
        default="",
        help=(
            "Comma-separated geo_filter codes for r/popular/best (e.g. us, ar, au)."
        ),
    )
    parser.add_argument(
        "--popular-geo-sorts",
        default="best,hot,new,top,rising",
        help=(
            "Comma-separated sorts for geo-filtered r/popular listings (choices: best, hot, new, top, rising)."
        ),
    )
    parser.add_argument(
        "--user",
        dest="users",
        action="append",
        default=[],
        help="Reddit username to scrape (overview, submitted, comments). Provide multiple times for multiple users.",
    )
    parser.add_argument(
        "--user-sections",
        default="overview,submitted,comments",
        help="Comma-separated user sections to scrape (choices: overview, submitted, comments).",
    )
    parser.add_argument(
        "--user-sorts",
        default="new,hot,top",
        help="Comma-separated sorts for user listings (choices: new, hot, top, best).",
    )
    parser.add_argument(
        "--listing-url",
        action="append",
        default=[],
        help="Additional listing JSON URL to scrape. Provide multiple times for multiple endpoints.",
    )
    parser.add_argument(
        "--post-url",
        action="append",
        default=[],
        help="Reddit post permalink to scrape individually (fetches comments). Provide multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root directory where scrape artifacts are saved. Defaults to ./scrapi_reddit_data "
            "(override with SCRAPI_REDDIT_OUTPUT_DIR)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    subreddits = _resolve_subreddits(args.subreddits, args.prompt)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    output_formats: set[str]
    if args.output_format == "both":
        output_formats = {"json", "csv"}
    elif args.output_format == "csv":
        output_formats = {"csv"}
    else:
        output_formats = {"json"}

    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        media_filters = normalize_media_filter_tokens(
            _parse_csv(args.media_filter) if args.media_filter else None
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    options = ScrapeOptions(
        output_root=output_root,
        listing_limit=args.limit,
        comment_limit=args.comment_limit,
        delay=args.delay,
        time_filter=args.time_filter,
        output_formats=output_formats,
        fetch_comments=args.fetch_comments,
        resume=args.resume,
        download_media=args.download_media,
        media_filters=media_filters,
    )

    subreddit_sorts = _parse_csv(args.subreddit_sorts, default="top")
    subreddit_top_times = _parse_csv(
        args.subreddit_top_times, default=args.time_filter
    )
    if "top" in subreddit_sorts and not subreddit_top_times:
        subreddit_top_times = [args.time_filter]

    popular_sorts = _parse_csv(args.popular_sorts)
    popular_geo_sorts = _parse_csv(args.popular_geo_sorts)
    popular_top_times = _parse_csv(args.popular_top_times, default=args.time_filter)
    if ("top" in popular_sorts or "top" in popular_geo_sorts) and not popular_top_times:
        popular_top_times = [args.time_filter]
    if "top" not in popular_sorts and "top" not in popular_geo_sorts:
        popular_top_times = []

    popular_geo = [code for code in _parse_csv(args.popular_geo) if code]
    user_sections = _parse_csv(args.user_sections)
    user_sorts = _parse_csv(args.user_sorts)

    listing_targets = _build_targets(
        subreddits,
        subreddit_sorts=subreddit_sorts,
        subreddit_top_times=subreddit_top_times,
        include_frontpage=args.frontpage,
        include_r_all=args.include_r_all,
        include_popular=args.popular,
        popular_sorts=popular_sorts,
        popular_top_times=popular_top_times,
        popular_geo=popular_geo,
        popular_geo_sorts=popular_geo_sorts,
        users=[name.strip() for name in args.users if name and name.strip()],
        user_sections=user_sections,
        user_sorts=user_sorts,
        listing_urls=args.listing_url,
    )

    post_urls = [url.strip() for url in args.post_url if url and url.strip()]
    post_targets = [_post_target_from_url(url) for url in post_urls]

    if not listing_targets and not post_targets:
        raise SystemExit(
            "No targets selected. Provide listing targets (subreddits, --popular/--frontpage/--include-r-all, --user, --listing-url) or --post-url."
        )

    if args.rebuild_from_json:
        if post_targets:
            print("Skipping --post-url targets during rebuild; only listings support CSV regeneration.", file=sys.stderr)
        for target in listing_targets:
            try:
                rebuild_csv_from_cache(target, options.output_root)
            except Exception as exc:  # noqa: BLE001 - surface error but continue
                print(f"Failed to rebuild CSV for {target.label}: {exc}", file=sys.stderr)
        return

    session = build_session(args.user_agent, not args.insecure)

    for target in listing_targets:
        try:
            process_listing(target, session=session, options=options)
        except Exception as exc:  # noqa: BLE001 - keep processing other subreddits
            print(f"Failed to process {target.label}: {exc}", file=sys.stderr)

    for target in post_targets:
        try:
            process_post(target, session=session, options=options)
        except Exception as exc:  # noqa: BLE001 - continue processing remaining posts
            print(f"Failed to process {target.label}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
