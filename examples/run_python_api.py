from __future__ import annotations

from pathlib import Path

from scrapi_reddit import ScrapeOptions, build_search_target, build_session, process_listing


def main() -> None:
    """Demonstrate the Python API by running a small search scrape."""
    session = build_session("scrapi-reddit-example/0.1", verify=True)

    options = ScrapeOptions(
        output_root=Path("./example_runs"),
        listing_limit=25,
        comment_limit=100,
        delay=1.5,
        time_filter="day",
        output_formats={"json", "csv"},
        fetch_comments=True,
        resume=True,
        download_media=False,
    )

    target = build_search_target(
        "python asyncio",
        search_types=["comment"],
        sort="new",
        time_filter="week",
    )

    process_listing(target, session=session, options=options)



if __name__ == "__main__":
    main()
