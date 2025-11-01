from __future__ import annotations

import json
from pathlib import Path

from scrapi_reddit.core import (
    BASE_URL,
    ListingTarget,
    ScrapeOptions,
    derive_filename,
    extract_links,
    rebuild_csv_from_cache,
)


def test_extract_links_parses_permalinks():
    listing_json = {
        "data": {
            "children": [
                {
                    "data": {
                        "id": "abc123",
                        "title": "Hello World",
                        "permalink": "r/test/comments/abc123/hello_world",
                        "created_utc": 1700000000,
                    }
                }
            ]
        }
    }

    links = extract_links(listing_json)

    assert len(links) == 1
    link = links[0]
    assert link["rank"] == 1
    assert link["url"].startswith(BASE_URL)
    assert link["url"].endswith(".json")


def test_derive_filename_includes_rank_and_slug():
    link_info = {
        "rank": 5,
        "id": "xy1",
        "title": "A neat post",
        "created_utc": 1700000000,
        "permalink": "/r/test/comments/xy1/a_neat_post/",
    }
    filename = derive_filename(link_info, post_data=None)

    assert filename.startswith("005_")
    assert filename.endswith(".json")
    assert "A_neat_post" in filename


def test_rebuild_csv_from_cache(tmp_path: Path):
    subreddit = "example"
    target = ListingTarget(
        label="r/example top (day)",
        output_segments=("subreddits", "example", "top_day"),
        url=f"{BASE_URL}/r/{subreddit}/top/.json",
        params={"t": "day"},
        context=subreddit,
    )
    base_dir = target.output_dir(tmp_path)
    posts_dir = base_dir / "post_jsons"
    posts_dir.mkdir(parents=True)

    listing_entry = {
        "kind": "Listing",
        "data": {
            "children": [
                {
                    "data": {
                        "id": "pq1",
                        "title": "Post title",
                        "created_utc": 1700000500,
                        "permalink": "/r/example/comments/pq1/post_title/",
                    }
                }
            ]
        },
    }
    comments_entry = {
        "kind": "Listing",
        "data": {
            "children": [
                {
                    "kind": "t1",
                    "data": {
                        "id": "c1",
                        "parent_id": "t3_pq1",
                        "author": "foo",
                        "body": "Nice",
                        "score": 1,
                        "created_utc": 1700000600,
                        "permalink": "/r/example/comments/pq1/post_title/c1/",
                    },
                }
            ]
        },
    }
    post_json_path = posts_dir / "001_post.json"
    post_json_path.write_text(json.dumps([listing_entry, comments_entry]), encoding="utf-8")

    links_path = base_dir / "links.json"
    links_path.write_text(
        json.dumps(
            [
                {
                    "rank": 1,
                    "id": "pq1",
                    "title": "Post title",
                    "created_utc": 1700000500,
                    "permalink": "/r/example/comments/pq1/post_title/",
                    "url": f"{BASE_URL}/r/example/comments/pq1/post_title/.json",
                }
            ]
        ),
        encoding="utf-8",
    )

    rebuild_csv_from_cache(target, tmp_path)

    posts_csv = (base_dir / "posts.csv").read_text(encoding="utf-8")
    comments_csv = (base_dir / "comments.csv").read_text(encoding="utf-8")

    assert "Post title" in posts_csv
    assert "c1" in comments_csv


def test_scrape_options_enforces_bounds(tmp_path: Path):
    options = ScrapeOptions(
        output_root=tmp_path,
        listing_limit=1000,
        comment_limit=9999,
        delay=0.1,
        time_filter="day",
        output_formats={"json"},
    )

    assert options.listing_limit == 100
    assert options.comment_limit == 500
    assert options.delay >= 1.0
