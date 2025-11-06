from __future__ import annotations

import json
from pathlib import Path

import pytest

import requests

import scrapi_reddit.core as core
from scrapi_reddit.core import (
    BASE_URL,
    ListingTarget,
    ScrapeOptions,
    derive_filename,
    extract_links,
    rebuild_csv_from_cache,
    normalize_media_filter_tokens,
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

    assert options.listing_limit == 1000
    assert options.comment_limit == 500
    assert options.delay >= 1.0
    assert options.listing_page_size == 100

    unlimited = ScrapeOptions(
        output_root=tmp_path,
        listing_limit=0,
        comment_limit=250,
        delay=2.0,
        time_filter="day",
        output_formats={"json"},
    )

    assert unlimited.listing_limit is None
    assert unlimited.comment_limit == 250

    zero_comment = ScrapeOptions(
        output_root=tmp_path,
        listing_limit=10,
        comment_limit=0,
        delay=2.0,
        time_filter="day",
        output_formats={"json"},
    )

    assert zero_comment.comment_limit == 500


def test_process_listing_resume_skips_fetch(tmp_path: Path, monkeypatch) -> None:
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

    listing_json = {
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
            ],
            "after": None,
        }
    }

    post_json = [
        {
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
            }
        },
        {
            "data": {"children": []}
        },
    ]

    cached_path = posts_dir / "001_pq1_cached.json"
    cached_path.write_text(json.dumps(post_json), encoding="utf-8")

    responses = [listing_json]

    def fake_fetch_json(session, url, *, params=None, retries=3, backoff=1.0):  # noqa: D401
        assert responses, "Unexpected additional fetch call"
        return responses.pop(0)

    monkeypatch.setattr(core, "fetch_json", fake_fetch_json)

    sleeps: list[float] = []
    monkeypatch.setattr(core.time, "sleep", lambda seconds: sleeps.append(seconds))

    options = ScrapeOptions(
        output_root=tmp_path,
        listing_limit=10,
        comment_limit=250,
        delay=1.0,
        time_filter="day",
        output_formats={"json"},
        fetch_comments=True,
        resume=True,
    )

    core.process_listing(target, session=object(), options=options)

    assert responses == []
    # Only the post delay should run (no rate limit waits, no fetch delays)
    assert sleeps == []


def test_process_listing_rate_limit_retries(tmp_path: Path, monkeypatch) -> None:
    subreddit = "example"
    target = ListingTarget(
        label="r/example hot",
        output_segments=("subreddits", "example", "hot"),
        url=f"{BASE_URL}/r/{subreddit}/hot/.json",
        context=subreddit,
    )

    base_dir = target.output_dir(tmp_path)
    (base_dir / "post_jsons").mkdir(parents=True)

    listing_json = {
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
            ],
            "after": None,
        }
    }

    post_json = [
        {
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
            }
        },
        {
            "data": {"children": []}
        },
    ]

    call_state = {"count": 0}

    def fake_fetch_json(session, url, *, params=None, retries=3, backoff=1.0):  # noqa: D401
        if "hot" in url and call_state["count"] == 0:
            call_state["count"] += 1
            return listing_json
        # Subsequent calls are for the post JSON
        post_call = call_state.setdefault("post_calls", 0)
        if post_call == 0:
            call_state["post_calls"] = 1
            raise RuntimeError("HTTP 429 Too Many Requests")
        if post_call == 1:
            call_state["post_calls"] = 2
            raise RuntimeError("429 second wave")
        return post_json

    monkeypatch.setattr(core, "fetch_json", fake_fetch_json)

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(core.time, "sleep", fake_sleep)

    options = ScrapeOptions(
        output_root=tmp_path,
        listing_limit=10,
        comment_limit=250,
        delay=0.1,
        time_filter="day",
        output_formats={"json"},
        fetch_comments=True,
    )

    core.process_listing(target, session=object(), options=options)

    # First rate limit should wait 60s, second should wait 120s before succeeding
    assert sleeps[:2] == [60, 120]


def test_collect_media_urls_pulls_multiple_sources() -> None:
    link_info = {
        "content_url": "https://www.reddit.com/r/example/comments/abc123/post/.json",
    }
    child_data = {
        "url_overridden_by_dest": "https://i.redd.it/example.png",
        "preview": {
            "images": [
                {
                    "source": {"url": "https://preview.redd.it/image.jpg"},
                    "variants": {
                        "gif": {"source": {"url": "https://preview.redd.it/image.gif"}},
                        "mp4": {"source": {"url": "https://preview.redd.it/image.mp4"}},
                    },
                }
            ]
        },
        "media_metadata": {
            "abc": {
                "s": {"u": "https://i.redd.it/gallery1.jpg"},
            }
        },
        "gallery_data": {
            "items": [
                {"media_id": "abc"},
            ]
        },
    }
    post_data = {
        "secure_media": {
            "reddit_video": {
                "fallback_url": "https://v.redd.it/video.mp4",
            }
        }
    }

    urls = core._collect_media_urls(link_info, child_data, post_data)

    assert "https://i.redd.it/example.png" in urls
    assert len(urls) == 4
    assert "https://i.redd.it/example.png" in urls
    assert "https://preview.redd.it/image.mp4" in urls
    assert "https://i.redd.it/gallery1.jpg" in urls
    assert "https://v.redd.it/video.mp4" in urls
    assert all("poster.jpg" not in url for url in urls)


def test_download_media_items_saves_files(tmp_path: Path) -> None:
    class FakeResponse:
        def __init__(self, *, headers: dict[str, str], payload: bytes, status_code: int = 200) -> None:
            self.headers = headers
            self._payload = payload
            self.status_code = status_code
            self.closed = False

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(str(self.status_code))

        def iter_content(self, chunk_size: int = 8192):  # noqa: D401 - generator helper
            yield self._payload

        def close(self) -> None:  # pragma: no cover - defensive
            self.closed = True

    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get(self, url: str, *, stream: bool, timeout: int):  # noqa: D401 - signature matches requests
            self.calls.append(url)
            if url.endswith("example.png"):
                return FakeResponse(headers={"Content-Type": "image/png"}, payload=b"png-bytes")
            return FakeResponse(headers={"Content-Type": "video/mp4"}, payload=b"mp4-bytes")

    session = FakeSession()
    urls = [
        "https://i.redd.it/example.png",
        "https://v.redd.it/example",
    ]

    saved = core._download_media_items(
        session,
        urls,
        media_dir=tmp_path,
        base_name="post",
        downloaded_urls=set(),
        resume=False,
    )

    assert saved == 2
    saved_files = sorted(p.name for p in tmp_path.iterdir())
    assert saved_files[0].startswith("post_media01")
    assert saved_files[1].startswith("post_media02")
    assert session.calls == urls


def test_collect_media_urls_skips_poster_thumbnails() -> None:
    link_info = {"content_url": "https://media.redgifs.com/example-poster.jpg"}
    urls = core._collect_media_urls(link_info, None, None)

    assert urls == []


def test_collect_media_urls_converts_gifv_to_mp4() -> None:
    gifv_url = "https://i.imgur.com/Example.gifv"
    link_info = {"content_url": gifv_url}
    child_data = {"url": gifv_url}

    urls = core._collect_media_urls(link_info, child_data, None)

    assert urls == ["https://i.imgur.com/Example.mp4"]


def test_collect_media_urls_prefers_video_over_duplicate_gif() -> None:
    link_info = {"content_url": "https://i.imgur.com/demo.gif"}
    child_data = {
        "url_overridden_by_dest": "https://i.imgur.com/demo.gifv",
        "preview": {
            "images": [
                {
                    "variants": {
                        "mp4": {"source": {"url": "https://i.imgur.com/demo.mp4"}},
                        "gif": {"source": {"url": "https://i.imgur.com/demo.gif"}},
                    }
                }
            ]
        },
    }

    urls = core._collect_media_urls(link_info, child_data, None)

    assert urls == ["https://i.imgur.com/demo.mp4"]


def test_collect_media_urls_adds_reddit_audio_track() -> None:
    fallback = "https://v.redd.it/example/DASH_720.mp4?source=fallback"
    link_info = {"content_url": fallback}
    child_data = {
        "secure_media": {
            "reddit_video": {
                "fallback_url": fallback,
            }
        }
    }

    urls = core._collect_media_urls(link_info, child_data, None)

    assert fallback in urls
    audio_url = core._derive_reddit_audio_url(fallback)
    assert audio_url in urls


def test_normalize_media_filter_tokens_accepts_categories_and_extensions() -> None:
    tokens = normalize_media_filter_tokens(["video", "mp4", "gif"])
    assert tokens == {"video", ".mp4", ".gif"}


def test_normalize_media_filter_tokens_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        normalize_media_filter_tokens(["unknown"])


def test_download_media_items_respects_manifest_on_resume(tmp_path: Path) -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get(self, url: str, *, stream: bool, timeout: int):  # noqa: D401 - signature compatibility
            self.calls.append(url)
            raise AssertionError("Download should be skipped when manifest already records URL")

    manifest_path = tmp_path / "media_manifest.json"
    manifest = {"https://i.redd.it/example.png": "post_media01.png"}

    saved = core._download_media_items(
        FakeSession(),
        ["https://i.redd.it/example.png"],
        media_dir=tmp_path,
        base_name="post",
        downloaded_urls=set(),
        resume=True,
        manifest=manifest,
        manifest_path=manifest_path,
    )

    assert saved == 0
    assert not manifest_path.exists()


def test_download_media_items_applies_filters(tmp_path: Path) -> None:
    class FakeResponse:
        def __init__(self, *, headers: dict[str, str], payload: bytes) -> None:
            self.headers = headers
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int = 8192):  # noqa: D401
            yield self._payload

        def close(self) -> None:  # pragma: no cover - interface compliance
            return None

    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get(self, url: str, *, stream: bool, timeout: int):  # noqa: D401
            self.calls.append(url)
            if url.endswith(".mp4"):
                return FakeResponse(headers={"Content-Type": "video/mp4"}, payload=b"mp4")
            if url.endswith(".jpg"):
                return FakeResponse(headers={"Content-Type": "image/jpeg"}, payload=b"jpg")
            raise AssertionError(f"Unexpected URL {url}")

    session = FakeSession()
    urls = [
        "https://i.redd.it/example.mp4",
        "https://i.redd.it/example.jpg",
    ]

    saved = core._download_media_items(
        session,
        urls,
        media_dir=tmp_path,
        base_name="post",
        downloaded_urls=set(),
        resume=False,
        allowed_filters={".mp4"},
    )

    assert saved == 1
    assert session.calls == ["https://i.redd.it/example.mp4"]
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".mp4"


def test_should_download_media_recognizes_audio_category() -> None:
    audio_url = "https://v.redd.it/example/DASH_audio.mp4"
    video_url = "https://v.redd.it/example/DASH_720.mp4"

    assert core._should_download_media(audio_url, {"audio"})
    assert not core._should_download_media(video_url, {"audio"})
