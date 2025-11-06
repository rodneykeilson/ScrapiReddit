# Scrapi Reddit

Scrapi Reddit is a zero-auth toolkit for scraping public Reddit listings. Use the CLI for quick data pulls or import the library to integrate pagination, comment harvesting, and CSV exports into your own workflows. This scraper fetches data from Reddit's Public API and does not require any API key.

## Features
- Scrape subreddit listings, the front page, r/popular (geo-aware), r/all, user activity, or custom listing URLs without OAuth.
- Toggle comment collection per post with resumable runs that reuse cached JSON and persist to CSV.
- Target individual posts to download full comment trees on demand.
- Automatic pagination, exponential backoff for rate limits, and structured logging with adjustable verbosity.
- Optional media capture downloads linked images, GIFs, and videos alongside post metadata.
- Media filters let you keep only the assets you need (e.g., videos only or static images only).
- Save outputs as JSON and optionally flatten posts/comments into CSV for downstream analysis.
- Configurable CLI plus Python API for scripting and integration.

## Important Notes
- Respect Reddit's [User Agreement](https://www.redditinc.com/policies/user-agreement) and local laws. Scraped data may have legal or ethical constraints.
- Heavy scraping can trigger rate limits or temporary IP bans. Provide a descriptive User-Agent and keep delays reasonable (I recommend 3 or 4 seconds delay).

## Dependencies
- Python 3.9+
- `requests` (runtime)
- `pytest` (tests, optional)

## Installation
```bash
pip install scrapi-reddit
```
After installation the console entry point `scrapi-reddit` is available on your PATH.

## Quick Start (CLI)
```bash
scrapi-reddit python --limit 200 --fetch-comments --output-format both
```
This command downloads up to 200 posts from r/python, fetches comments (up to 500 per post), and writes JSON + CSV outputs under `./scrapi_reddit_data`.

### Common CLI Options
- `--fetch-comments` Enable post-level comment requests (defaults off).
- `--comment-limit 0` Request the maximum 500 comments per post.
- `--continue` Resume a previous run by reusing cached post JSON files and skipping previously downloaded media.
- `--media-filter video,gif` Restrict downloads to specific categories or extensions (`video`, `image`, `animated`, `audio`, or extensions such as `mp4`, `jpg`, `gif`).
- `--download-media` Save linked images/GIFs/videos under each target's media directory.
- `--popular --popular-geo <region-code>` Pull popular listings with geo filters.
- `--user <name>` Scrape user overview/submitted/comments sections.

### Advanced CLI Examples
Fetch multiple subreddits with varied sorts and time windows:
```powershell
scrapi-reddit python typescript --subreddit-sorts top,hot --subreddit-top-times day,all --limit 500 --output-format both
```
Resume a long run after interruption:
```powershell
scrapi-reddit python --fetch-comments --continue --limit 1000 --log-level INFO
```
Download a single post (JSON + CSV):
```powershell
scrapi-reddit --post-url https://www.reddit.com/r/python/comments/xyz789/example_post/
```

## Python API
Import the library when you need finer control inside Python scripts.

### Step 1 – Configure a session
```python
from scrapi_reddit import build_session

session = build_session("your-app-name/0.1", verify=True)
```

### Step 2 – Define scrape options
```python
from pathlib import Path
from scrapi_reddit import ScrapeOptions

options = ScrapeOptions(
    output_root=Path("./scrapes"),
    listing_limit=250,
    comment_limit=0,      # auto-expand to 500
    delay=3.0,
    time_filter="day",
    output_formats={"json", "csv"},
    fetch_comments=True,
    resume=True,          # reuse cached JSON/media on reruns
    download_media=True,
    media_filters={"video", ".mp4"},
)
```

### Step 3 – Scrape a listing
```python
from scrapi_reddit import ListingTarget, process_listing

target = ListingTarget(
    label="r/python top (day)",
    output_segments=("subreddits", "python", "top_day"),
    url="https://www.reddit.com/r/python/top/.json",
    params={"t": "day"},
    context="python",
)

process_listing(target, session=session, options=options)
```

### Step 4 – Scrape a single post
```python
from scrapi_reddit import PostTarget, process_post

post_target = PostTarget(
    label="Example post",
    output_segments=("posts", "python", "xyz789"),
    url="https://www.reddit.com/r/python/comments/xyz789/example_post/.json",
)

process_post(post_target, session=session, options=options)
```
Both helpers write JSON/CSV to the configured output directory and emit progress via logging.
When `download_media=True` (or `--download-media` on the CLI) any discoverable images, GIFs, and videos are saved under a `media/` directory per target. Reddit hosts video and audio streams separately, so clips from `v.redd.it` arrive as two files (for example `*_media01.mp4` and `*_media01_audio.mp4`). Merge them with a tool like `ffmpeg -i video.mp4 -i video_audio.mp4 -c copy merged.mp4` if you need the original audio track inline.

## Testing
```bash
python -m pytest
```

## Contributing
Bug reports and pull requests are welcome. For feature requests or questions, please open an issue. When contributing, add tests that cover new behavior and ensure `python -m pytest` passes before submitting a PR.

## License
Released under the [MIT License](LICENSE). You may use, modify, and distribute this project with attribution and a copy of the license. Use at your own risk.