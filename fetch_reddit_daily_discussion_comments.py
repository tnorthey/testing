#!/usr/bin/env python3
"""
Fetch comments from a Reddit daily discussion thread.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


REDDIT_API_BASE = "https://www.reddit.com"
DEFAULT_QUERIES = "Daily Discussion,Daily General Discussion"
DEFAULT_USER_AGENT = "crypto-price-correlation-script (reddit-daily-comments)"
POST_ID_RE = re.compile(r"/comments/([a-z0-9]+)/")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z']{2,}")
TICKER_RE = re.compile(r"\$[A-Za-z]{2,6}")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
ETHEREUM_ONLY_TITLE_RE = re.compile(r"^\s*ethereum[!.]?\s*$", re.IGNORECASE)
ETHEREUM_FUZZY_BODY_RE = re.compile(
    r"^\s*[^A-Za-z]*e[^A-Za-z]*t[^A-Za-z]*h[^A-Za-z]*e[^A-Za-z]*r[^A-Za-z]*e[^A-Za-z]*u[^A-Za-z]*m[^A-Za-z]*\s*$",
    re.IGNORECASE,
)
STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "and",
    "any",
    "are",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "here",
    "how",
    "into",
    "its",
    "just",
    "more",
    "most",
    "not",
    "now",
    "off",
    "once",
    "only",
    "other",
    "out",
    "over",
    "same",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "too",
    "under",
    "until",
    "very",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class RedditPost:
    post_id: str
    title: str
    created_utc: int
    permalink: str
    num_comments: int
    score: int

    @property
    def url(self) -> str:
        return f"{REDDIT_API_BASE}{self.permalink}"


@dataclass(frozen=True)
class RedditComment:
    comment_id: str
    parent_id: str
    link_id: str
    author: str
    body: str
    created_utc: int
    score: int
    depth: int
    permalink: str

    @property
    def url(self) -> str:
        return f"{REDDIT_API_BASE}{self.permalink}"


def parse_queries(value: str) -> list[str]:
    queries = [item.strip() for item in value.split(",") if item.strip()]
    if not queries:
        raise ValueError("No search queries provided.")
    return queries


def parse_utc_date(value: str) -> dt.date:
    try:
        parsed = dt.date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc
    return parsed


def utc_today() -> dt.date:
    return dt.datetime.now(tz=dt.timezone.utc).date()


def utc_date_from_ts(ts: int) -> dt.date:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date()


def safe_filename(value: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", value.strip())
    return cleaned.strip("._-") or "value"


def cache_path_for_post(cache_dir: Path, subreddit: str, post_id: str) -> Path:
    return cache_dir / f"{safe_filename(subreddit)}_post_{safe_filename(post_id)}.json"


def cache_path_for_date(cache_dir: Path, subreddit: str, target_date: dt.date) -> Path:
    return cache_dir / f"{safe_filename(subreddit)}_daily_{target_date.isoformat()}.json"


def load_cached_daily(path: Path) -> Optional[tuple[RedditPost, list[dict[str, object]]]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    post_data = payload.get("post")
    comments_data = payload.get("comments")
    if not isinstance(post_data, dict) or not isinstance(comments_data, list):
        return None

    try:
        post = RedditPost(
            post_id=str(post_data["post_id"]),
            title=str(post_data["title"]),
            created_utc=int(post_data["created_utc"]),
            permalink=str(post_data["permalink"]),
            num_comments=int(post_data.get("num_comments", len(comments_data))),
            score=int(post_data.get("score", 0)),
        )
    except (KeyError, TypeError, ValueError):
        return None

    records: list[dict[str, object]] = []
    for item in comments_data:
        if isinstance(item, dict):
            records.append(item)
    return post, records


def save_cached_daily(
    path: Path,
    *,
    subreddit: str,
    post: RedditPost,
    records: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "fetched_at_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "subreddit": subreddit,
        "post": {
            "post_id": post.post_id,
            "title": post.title,
            "created_utc": post.created_utc,
            "permalink": post.permalink,
            "num_comments": post.num_comments,
            "score": post.score,
            "url": post.url,
        },
        "comments": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch_json(url: str, user_agent: str, timeout: int) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Reddit request failed: {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Reddit request failed: {exc.reason}") from exc
    return json.loads(payload)


def build_search_url(
    subreddit: str,
    query: str,
    sort: str,
    time_filter: str,
    limit: int,
    after: Optional[str] = None,
) -> str:
    params = {
        "q": query,
        "restrict_sr": "1",
        "sort": sort,
        "t": time_filter,
        "limit": str(limit),
        "raw_json": "1",
    }
    if after:
        params["after"] = after
    return f"{REDDIT_API_BASE}/r/{subreddit}/search.json?{urllib.parse.urlencode(params)}"


def parse_posts(payload: object) -> list[RedditPost]:
    if not isinstance(payload, dict):
        return []
    children = payload.get("data", {}).get("children", [])
    posts: list[RedditPost] = []
    for child in children:
        if child.get("kind") != "t3":
            continue
        data = child.get("data", {})
        post_id = data.get("id")
        title = data.get("title")
        created_utc = data.get("created_utc")
        permalink = data.get("permalink")
        num_comments = data.get("num_comments")
        score = data.get("score", 0)
        if not all([post_id, title, created_utc, permalink, num_comments is not None]):
            continue
        posts.append(
            RedditPost(
                post_id=str(post_id),
                title=str(title),
                created_utc=int(created_utc),
                permalink=str(permalink),
                num_comments=int(num_comments),
                score=int(score) if score is not None else 0,
            )
        )
    return posts


def title_matches_queries(title: str, queries: Iterable[str]) -> bool:
    title_lower = title.lower()
    return any(query.lower() in title_lower for query in queries)


def is_ethereum_only_title(title: str) -> bool:
    return bool(ETHEREUM_ONLY_TITLE_RE.match(title))


def is_ethereum_only_body(body: str) -> bool:
    return bool(ETHEREUM_FUZZY_BODY_RE.match(body))


def best_ethereum_comment_from_records(
    records: Iterable[dict[str, object]],
) -> Optional[tuple[dict[str, object], int]]:
    best: Optional[dict[str, object]] = None
    best_score: Optional[int] = None

    for record in records:
        body = record.get("body", "")
        if not isinstance(body, str) or not is_ethereum_only_body(body):
            continue
        score_val = record.get("score", 0)
        try:
            score = int(score_val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            score = 0
        if best is None or best_score is None or score > best_score:
            best = record
            best_score = score

    if best is None or best_score is None:
        return None
    return best, best_score


def index_cached_posts_by_date(
    cache_dir: Path,
    *,
    subreddit: str,
    queries: list[str],
) -> dict[dt.date, tuple[RedditPost, list[dict[str, object]]]]:
    pattern = f"{safe_filename(subreddit)}_post_*.json"
    best_for_date: dict[dt.date, tuple[RedditPost, list[dict[str, object]]]] = {}

    for path in cache_dir.glob(pattern):
        loaded = load_cached_daily(path)
        if loaded is None:
            continue
        post, records = loaded
        # Prefer only daily discussion thread caches, to avoid mixing in unrelated posts.
        if not title_matches_queries(post.title, queries):
            continue
        date_value = utc_date_from_ts(post.created_utc)
        existing = best_for_date.get(date_value)
        if existing is None or post.created_utc > existing[0].created_utc:
            best_for_date[date_value] = (post, records)

    return best_for_date


def search_posts_paginated(
    *,
    subreddit: str,
    query: str,
    sort: str,
    time_filter: str,
    min_date_inclusive: dt.date,
    user_agent: str,
    timeout: int,
    per_page: int = 100,
    max_pages: int = 25,
) -> list[RedditPost]:
    per_page = max(1, min(100, per_page))
    after: Optional[str] = None
    posts: list[RedditPost] = []

    for _ in range(max_pages):
        url = build_search_url(
            subreddit=subreddit,
            query=query,
            sort=sort,
            time_filter=time_filter,
            limit=per_page,
            after=after,
        )
        payload = fetch_json(url, user_agent, timeout)
        page_posts = parse_posts(payload)
        if not page_posts:
            break
        posts.extend(page_posts)

        oldest_date = min(utc_date_from_ts(post.created_utc) for post in page_posts)
        after = None
        if isinstance(payload, dict):
            after_value = payload.get("data", {}).get("after")
            if isinstance(after_value, str) and after_value:
                after = after_value

        if after is None:
            break
        if oldest_date < min_date_inclusive:
            break

    return posts


def build_listing_url(
    subreddit: str,
    listing: str,
    limit: int,
    after: Optional[str] = None,
) -> str:
    params = {
        "limit": str(limit),
        "raw_json": "1",
    }
    if after:
        params["after"] = after
    return f"{REDDIT_API_BASE}/r/{subreddit}/{listing}.json?{urllib.parse.urlencode(params)}"


def list_posts_paginated(
    *,
    subreddit: str,
    listing: str,
    min_date_inclusive: dt.date,
    user_agent: str,
    timeout: int,
    per_page: int = 100,
    max_pages: int = 200,
) -> list[RedditPost]:
    per_page = max(1, min(100, per_page))
    after: Optional[str] = None
    posts: list[RedditPost] = []

    for _ in range(max_pages):
        url = build_listing_url(
            subreddit=subreddit,
            listing=listing,
            limit=per_page,
            after=after,
        )
        payload = fetch_json(url, user_agent, timeout)
        page_posts = parse_posts(payload)
        if not page_posts:
            break
        posts.extend(page_posts)

        oldest_date = min(utc_date_from_ts(post.created_utc) for post in page_posts)

        after = None
        if isinstance(payload, dict):
            after_value = payload.get("data", {}).get("after")
            if isinstance(after_value, str) and after_value:
                after = after_value

        if after is None:
            break
        if oldest_date < min_date_inclusive:
            break

    return posts

def pick_latest_post(posts: Iterable[RedditPost], queries: list[str]) -> Optional[RedditPost]:
    posts_list = list(posts)
    if not posts_list:
        return None
    matches = [post for post in posts_list if title_matches_queries(post.title, queries)]
    if not matches:
        return max(posts_list, key=lambda post: post.created_utc)
    return max(matches, key=lambda post: post.created_utc)


def pick_post_for_date(
    posts: Iterable[RedditPost],
    queries: list[str],
    target_date: dt.date,
) -> Optional[RedditPost]:
    posts_list = [post for post in posts if title_matches_queries(post.title, queries)]
    if not posts_list:
        return None

    same_day = [post for post in posts_list if utc_date_from_ts(post.created_utc) == target_date]
    if same_day:
        return max(same_day, key=lambda post: post.created_utc)

    # If we can't find an exact match, pick the closest in time so the script still works.
    # This can happen if the post was published near midnight UTC or search results are limited.
    target_dt = dt.datetime.combine(target_date, dt.time(12, 0), tzinfo=dt.timezone.utc)
    return min(
        posts_list,
        key=lambda post: abs(
            dt.datetime.fromtimestamp(post.created_utc, tz=dt.timezone.utc) - target_dt
        ),
    )


def extract_post_id(post_url: str) -> str:
    match = POST_ID_RE.search(post_url)
    if not match:
        raise ValueError(f"Unable to extract post id from URL: {post_url}")
    return match.group(1)


def build_comments_url(post_id: str, sort: str, limit: int) -> str:
    params = {
        "sort": sort,
        "limit": str(limit),
        "raw_json": "1",
    }
    return f"{REDDIT_API_BASE}/comments/{post_id}.json?{urllib.parse.urlencode(params)}"


def parse_post_from_listing(payload: object) -> Optional[RedditPost]:
    if not isinstance(payload, dict):
        return None
    children = payload.get("data", {}).get("children", [])
    for child in children:
        if child.get("kind") != "t3":
            continue
        data = child.get("data", {})
        post_id = data.get("id")
        title = data.get("title")
        created_utc = data.get("created_utc")
        permalink = data.get("permalink")
        num_comments = data.get("num_comments")
        score = data.get("score", 0)
        if not all([post_id, title, created_utc, permalink, num_comments is not None]):
            continue
        return RedditPost(
            post_id=str(post_id),
            title=str(title),
            created_utc=int(created_utc),
            permalink=str(permalink),
            num_comments=int(num_comments),
            score=int(score) if score is not None else 0,
        )
    return None


def extract_comments(
    payload: object,
    include_deleted: bool,
    min_score: Optional[int],
    max_comments: Optional[int],
) -> list[RedditComment]:
    if not isinstance(payload, dict):
        return []
    children = payload.get("data", {}).get("children", [])
    comments: list[RedditComment] = []

    def should_skip(author: str, body: str, score: int) -> bool:
        if min_score is not None and score < min_score:
            return True
        if include_deleted:
            return False
        if not author or author == "[deleted]":
            return True
        if body in ("[deleted]", "[removed]"):
            return True
        return False

    def walk(nodes: list[object]) -> bool:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if node.get("kind") != "t1":
                continue
            data = node.get("data", {})
            comment_id = data.get("id")
            parent_id = data.get("parent_id")
            link_id = data.get("link_id")
            author = data.get("author", "")
            body = data.get("body", "")
            created_utc = data.get("created_utc")
            score = data.get("score", 0)
            depth = data.get("depth", 0)
            permalink = data.get("permalink", "")
            if not all([comment_id, parent_id, link_id, created_utc, permalink]):
                continue
            if should_skip(str(author), str(body), int(score)):
                pass
            else:
                comments.append(
                    RedditComment(
                        comment_id=str(comment_id),
                        parent_id=str(parent_id),
                        link_id=str(link_id),
                        author=str(author),
                        body=str(body),
                        created_utc=int(created_utc),
                        score=int(score),
                        depth=int(depth),
                        permalink=str(permalink),
                    )
                )
                if max_comments and len(comments) >= max_comments:
                    return True

            replies = data.get("replies")
            if isinstance(replies, dict):
                reply_children = replies.get("data", {}).get("children", [])
                if walk(reply_children):
                    return True
        return False

    walk(children)
    return comments


def format_timestamp(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat()


def to_ascii(text: str) -> str:
    return text.encode("ascii", "ignore").decode("ascii")


def truncate_text(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    cleaned = to_ascii(cleaned)
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 3:
        return cleaned[:limit]
    return cleaned[: limit - 3].rstrip() + "..."


def summarize_comments(
    comments: Iterable[RedditComment],
    post: RedditPost,
    max_terms: int = 10,
    max_authors: int = 5,
    max_examples: int = 3,
) -> str:
    comments_list = list(comments)
    if not comments_list:
        return "No comments available to summarize."

    author_counts = Counter(comment.author for comment in comments_list)
    word_counts: Counter[str] = Counter()
    ticker_counts: Counter[str] = Counter()
    timestamps = [comment.created_utc for comment in comments_list]

    for comment in comments_list:
        body = comment.body
        for match in WORD_RE.finditer(body.lower()):
            word = match.group(0)
            if word in STOPWORDS:
                continue
            word_counts[word] += 1
        for match in TICKER_RE.finditer(body):
            ticker_counts[match.group(0).upper()] += 1

    top_terms = [f"{term} ({count})" for term, count in word_counts.most_common(max_terms)]
    top_authors = [
        f"{author} ({count})" for author, count in author_counts.most_common(max_authors)
    ]
    top_tickers = [
        f"{ticker} ({count})" for ticker, count in ticker_counts.most_common(5)
    ]
    top_scored = sorted(comments_list, key=lambda comment: comment.score, reverse=True)[
        :max_examples
    ]

    lines = [
        f"Summary for '{to_ascii(post.title)}'",
        f"Post URL: {post.url}",
        f"Comments: {len(comments_list)} | Unique authors: {len(author_counts)}",
        (
            "Time range (UTC): "
            f"{format_timestamp(min(timestamps))} to {format_timestamp(max(timestamps))}"
        ),
    ]

    if top_tickers:
        lines.append("Top tickers: " + ", ".join(top_tickers))
    if top_terms:
        lines.append("Top keywords: " + ", ".join(top_terms))
    if top_authors:
        lines.append("Top commenters: " + ", ".join(top_authors))
    if top_scored:
        lines.append("Top scored comments:")
        for comment in top_scored:
            snippet = truncate_text(comment.body, 160)
            lines.append(f"- ({comment.score}) {snippet}")

    return "\n".join(lines)


def summarize_comment_records(records: Iterable[dict[str, object]], post: RedditPost) -> str:
    # Use the same summary logic, but with cached/normalized records instead of RedditComment.
    comments_list: list[dict[str, object]] = []
    for record in records:
        if isinstance(record, dict):
            comments_list.append(record)

    if not comments_list:
        return "No comments available to summarize."

    def get_str(item: dict[str, object], key: str) -> str:
        value = item.get(key, "")
        return value if isinstance(value, str) else str(value)

    def get_int(item: dict[str, object], key: str, default: int = 0) -> int:
        value = item.get(key, default)
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    author_counts = Counter(get_str(comment, "author") for comment in comments_list)
    word_counts: Counter[str] = Counter()
    ticker_counts: Counter[str] = Counter()
    timestamps = [get_int(comment, "created_utc", 0) for comment in comments_list if get_int(comment, "created_utc", 0)]

    for comment in comments_list:
        body = get_str(comment, "body")
        for match in WORD_RE.finditer(body.lower()):
            word = match.group(0)
            if word in STOPWORDS:
                continue
            word_counts[word] += 1
        for match in TICKER_RE.finditer(body):
            ticker_counts[match.group(0).upper()] += 1

    top_terms = [f"{term} ({count})" for term, count in word_counts.most_common(10)]
    top_authors = [f"{author} ({count})" for author, count in author_counts.most_common(5)]
    top_tickers = [f"{ticker} ({count})" for ticker, count in ticker_counts.most_common(5)]
    top_scored = sorted(comments_list, key=lambda comment: get_int(comment, "score", 0), reverse=True)[:3]

    if timestamps:
        time_range = (
            "Time range (UTC): "
            f"{format_timestamp(min(timestamps))} to {format_timestamp(max(timestamps))}"
        )
    else:
        time_range = "Time range (UTC): unknown"

    lines = [
        f"Summary for '{to_ascii(post.title)}'",
        f"Post URL: {post.url}",
        f"Comments: {len(comments_list)} | Unique authors: {len(author_counts)}",
        time_range,
    ]

    if top_tickers:
        lines.append("Top tickers: " + ", ".join(top_tickers))
    if top_terms:
        lines.append("Top keywords: " + ", ".join(top_terms))
    if top_authors:
        lines.append("Top commenters: " + ", ".join(top_authors))
    if top_scored:
        lines.append("Top scored comments:")
        for comment in top_scored:
            snippet = truncate_text(get_str(comment, "body"), 160)
            lines.append(f"- ({get_int(comment, 'score', 0)}) {snippet}")

    return "\n".join(lines)


def write_jsonl(
    comments: Iterable[RedditComment],
    post: RedditPost,
    handle: object,
) -> None:
    for record in iter_comment_records(comments, post):
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def iter_comment_records(
    comments: Iterable[RedditComment],
    post: RedditPost,
) -> Iterable[dict[str, object]]:
    for comment in comments:
        yield {
            "post_id": post.post_id,
            "post_title": post.title,
            "post_url": post.url,
            "post_created_utc": post.created_utc,
            "post_created_iso": format_timestamp(post.created_utc),
            "comment_id": comment.comment_id,
            "comment_url": comment.url,
            "parent_id": comment.parent_id,
            "link_id": comment.link_id,
            "author": comment.author,
            "body": comment.body,
            "created_utc": comment.created_utc,
            "created_iso": format_timestamp(comment.created_utc),
            "score": comment.score,
            "depth": comment.depth,
        }


def write_json(
    comments: Iterable[RedditComment],
    post: RedditPost,
    handle: object,
    *,
    pretty: bool = False,
) -> None:
    records = list(iter_comment_records(comments, post))
    if pretty:
        handle.write(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    else:
        handle.write(json.dumps(records, ensure_ascii=True) + "\n")


def write_jsonl_records(records: Iterable[dict[str, object]], handle: object) -> None:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_json_records(
    records: list[dict[str, object]],
    handle: object,
    *,
    pretty: bool = False,
) -> None:
    if pretty:
        handle.write(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    else:
        handle.write(json.dumps(records, ensure_ascii=True) + "\n")


def write_csv_records(records: Iterable[dict[str, object]], handle: object) -> None:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "post_id",
            "post_title",
            "post_url",
            "post_created_utc",
            "post_created_iso",
            "comment_id",
            "comment_url",
            "parent_id",
            "link_id",
            "author",
            "body",
            "created_utc",
            "created_iso",
            "score",
            "depth",
        ]
    )
    for record in records:
        writer.writerow(
            [
                record.get("post_id", ""),
                record.get("post_title", ""),
                record.get("post_url", ""),
                record.get("post_created_utc", ""),
                record.get("post_created_iso", ""),
                record.get("comment_id", ""),
                record.get("comment_url", ""),
                record.get("parent_id", ""),
                record.get("link_id", ""),
                record.get("author", ""),
                record.get("body", ""),
                record.get("created_utc", ""),
                record.get("created_iso", ""),
                record.get("score", ""),
                record.get("depth", ""),
            ]
        )


def write_csv(
    comments: Iterable[RedditComment],
    post: RedditPost,
    handle: object,
) -> None:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "post_id",
            "post_title",
            "post_url",
            "post_created_utc",
            "post_created_iso",
            "comment_id",
            "comment_url",
            "parent_id",
            "link_id",
            "author",
            "body",
            "created_utc",
            "created_iso",
            "score",
            "depth",
        ]
    )
    for comment in comments:
        writer.writerow(
            [
                post.post_id,
                post.title,
                post.url,
                post.created_utc,
                format_timestamp(post.created_utc),
                comment.comment_id,
                comment.url,
                comment.parent_id,
                comment.link_id,
                comment.author,
                comment.body,
                comment.created_utc,
                format_timestamp(comment.created_utc),
                comment.score,
                comment.depth,
            ]
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Reddit comments from a daily discussion thread."
    )
    parser.add_argument(
        "--subreddit",
        default="ethereum",
        help="Target subreddit (default: ethereum).",
    )
    parser.add_argument(
        "--queries",
        default=DEFAULT_QUERIES,
        help="Comma-separated search queries (default: Daily Discussion,Daily General Discussion).",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--post-id",
        help="Explicit Reddit post id (skips search).",
    )
    source_group.add_argument(
        "--post-url",
        help="Explicit Reddit post URL (skips search).",
    )
    source_group.add_argument(
        "--date",
        type=parse_utc_date,
        help="UTC date (YYYY-MM-DD) of the daily discussion to fetch (uses search).",
    )
    source_group.add_argument(
        "--days-ago",
        type=int,
        help="Fetch the daily discussion from N days ago (UTC, uses search).",
    )
    source_group.add_argument(
        "--past-days",
        type=int,
        help="Look back over the last N days (UTC). Use with --summary-only or --ethereum-upvotes-csv.",
    )
    parser.add_argument(
        "--search-sort",
        default="new",
        choices=("relevance", "hot", "top", "new"),
        help="Search sort order (default: new).",
    )
    parser.add_argument(
        "--time-filter",
        default="day",
        choices=("hour", "day", "week", "month", "year", "all"),
        help="Search time filter (default: day).",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=10,
        help="Max search results per query (default: 10).",
    )
    parser.add_argument(
        "--comment-sort",
        default="new",
        choices=("confidence", "top", "new", "controversial", "old", "qa"),
        help="Comment sort order (default: new).",
    )
    parser.add_argument(
        "--comment-limit",
        type=int,
        default=500,
        help="Limit for comment listing request (default: 500).",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=0,
        help="Cap the number of comments returned (default: 0, no cap).",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        help="Minimum comment score to include.",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include deleted or removed comments.",
    )
    parser.add_argument(
        "--format",
        default="jsonl",
        choices=("jsonl", "json", "csv"),
        help="Output format (default: jsonl).",
    )
    parser.add_argument(
        "--output",
        help="Output path (default: stdout).",
    )
    parser.add_argument(
        "--ethereum-upvotes-csv",
        help=(
            "Write a CSV of UTC date vs upvotes for the best comment whose body is exactly "
            "'Ethereum' (or 'Ethereum!' / 'Ethereum.'; case-insensitive). Requires --past-days and --cache-dir."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory to read/write cached JSON (per post/date). If present, cached data is used instead of refetching.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore any cached files and refetch from Reddit.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a text summary to stderr after fetching comments.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print the summary to stdout (skip comment output).",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent string for Reddit requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds (default: 30).",
    )
    return parser


def collect_candidate_posts(args: argparse.Namespace) -> list[RedditPost]:
    queries = parse_queries(args.queries)
    posts: dict[str, RedditPost] = {}

    for query in queries:
        url = build_search_url(
            subreddit=args.subreddit,
            query=query,
            sort=args.search_sort,
            time_filter=args.time_filter,
            limit=args.search_limit,
        )
        payload = fetch_json(url, args.user_agent, args.timeout)
        for post in parse_posts(payload):
            posts[post.post_id] = post
    return list(posts.values())


def find_daily_post(args: argparse.Namespace, target_date: Optional[dt.date] = None) -> RedditPost:
    queries = parse_queries(args.queries)
    posts = collect_candidate_posts(args)

    if target_date is None:
        latest = pick_latest_post(posts, queries)
        if latest:
            return latest
        raise RuntimeError("No matching daily discussion thread found.")

    chosen = pick_post_for_date(posts, queries, target_date)
    if chosen:
        return chosen
    raise RuntimeError("No matching daily discussion thread found for the requested date.")


def find_past_daily_posts(args: argparse.Namespace, days: int) -> list[tuple[dt.date, RedditPost]]:
    queries = parse_queries(args.queries)
    posts = collect_candidate_posts(args)
    by_date: list[tuple[dt.date, RedditPost]] = []

    today = utc_today()
    for offset in range(days):
        target_date = today - dt.timedelta(days=offset)
        chosen = pick_post_for_date(posts, queries, target_date)
        if chosen is None:
            continue
        by_date.append((target_date, chosen))

    # Deduplicate in case multiple dates pick the same "closest" post.
    seen_ids: set[str] = set()
    unique: list[tuple[dt.date, RedditPost]] = []
    for date_value, post in by_date:
        if post.post_id in seen_ids:
            continue
        seen_ids.add(post.post_id)
        unique.append((date_value, post))

    return unique


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.days_ago is not None and args.days_ago < 0:
        print("Error: --days-ago must be >= 0.", file=sys.stderr)
        return 1
    if args.past_days is not None and args.past_days <= 0:
        print("Error: --past-days must be > 0.", file=sys.stderr)
        return 1
    if args.ethereum_upvotes_csv and args.past_days is None and args.date is None:
        print("Error: --ethereum-upvotes-csv requires --past-days or --date.", file=sys.stderr)
        return 1
    if args.past_days is not None and not (args.summary_only or args.ethereum_upvotes_csv):
        print("Error: --past-days requires --summary-only or --ethereum-upvotes-csv.", file=sys.stderr)
        return 1
    if args.ethereum_upvotes_csv and args.cache_dir is None:
        print("Error: --ethereum-upvotes-csv requires --cache-dir (containing cached ethereum_post_*.json files).", file=sys.stderr)
        return 1

    # For past dates, default search time filter 'day' is too restrictive.
    if (args.date is not None or args.days_ago is not None or args.past_days is not None) and (
        args.time_filter == "day"
    ):
        args.time_filter = "all"
    # Also bump default search limit so older posts are likely included.
    if (args.date is not None or args.days_ago is not None or args.past_days is not None) and (
        args.search_limit < 50
    ):
        args.search_limit = 50

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    refresh_cache = bool(args.refresh_cache)

    post_id = None
    post = None

    if args.ethereum_upvotes_csv:
        if args.date is not None:
            start_date = args.date
            end_date = args.date
        else:
            today = utc_today()
            start_date = today - dt.timedelta(days=int(args.past_days) - 1)
            end_date = today

        assert cache_dir is not None
        queries = parse_queries(args.queries)
        cached_by_date = index_cached_posts_by_date(cache_dir, subreddit=args.subreddit, queries=queries)

        output_handle = None
        try:
            if args.ethereum_upvotes_csv == "-":
                output_handle = sys.stdout
            else:
                out_path = Path(args.ethereum_upvotes_csv).expanduser()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                output_handle = open(out_path, "w", newline="", encoding="utf-8")

            writer = csv.writer(output_handle)
            writer.writerow(
                [
                    "date_utc",
                    "upvotes",
                    "comment_id",
                    "comment_url",
                    "comment_author",
                    "comment_body",
                    "post_id",
                    "post_url",
                ]
            )
            current = start_date
            while current <= end_date:
                cached = cached_by_date.get(current)
                if cached is None:
                    # Fall back to date-based cache if you used --date/--past-days caching.
                    loaded = load_cached_daily(cache_path_for_date(cache_dir, args.subreddit, current))
                    if loaded is not None:
                        cached = loaded

                if cached is None:
                    writer.writerow([current.isoformat(), "", "", "", "", "", "", ""])
                else:
                    post_for_day, records = cached
                    post_date = utc_date_from_ts(post_for_day.created_utc)
                    if post_date != current:
                        # Avoid duplicating the same post across dates if the cache is misaligned.
                        writer.writerow([current.isoformat(), "", "", "", "", "", "", ""])
                        current += dt.timedelta(days=1)
                        continue
                    best_comment = best_ethereum_comment_from_records(records)
                    if best_comment is None:
                        writer.writerow(
                            [
                                current.isoformat(),
                                "",
                                "",
                                "",
                                "",
                                "",
                                post_for_day.post_id,
                                post_for_day.url,
                            ]
                        )
                    else:
                        best_record, best_score = best_comment
                        writer.writerow(
                            [
                                current.isoformat(),
                                best_score,
                                best_record.get("comment_id", ""),
                                best_record.get("comment_url", ""),
                                best_record.get("author", ""),
                                best_record.get("body", ""),
                                post_for_day.post_id,
                                post_for_day.url,
                            ]
                        )
                current += dt.timedelta(days=1)
        finally:
            if output_handle is not None and output_handle is not sys.stdout:
                output_handle.close()

        return 0

    if args.past_days is not None:
        candidates: Optional[list[RedditPost]] = None
        queries = parse_queries(args.queries)
        today = utc_today()

        for offset in range(args.past_days - 1, -1, -1):
            date_value = today - dt.timedelta(days=offset)

            # 1) Try date-based cache first (no search required).
            if cache_dir is not None and not refresh_cache:
                cached = load_cached_daily(cache_path_for_date(cache_dir, args.subreddit, date_value))
                if cached is not None:
                    cached_post, cached_records = cached
                    print(f"=== {date_value.isoformat()} ===")
                    print(summarize_comment_records(cached_records, cached_post))
                    print()
                    continue

            # 2) Build candidate posts (search) once, only if needed.
            if candidates is None:
                candidates = collect_candidate_posts(args)
                if not candidates:
                    print("Error: No matching daily discussion threads found.", file=sys.stderr)
                    return 1

            chosen = pick_post_for_date(candidates, queries, date_value)
            if chosen is None:
                print(f"=== {date_value.isoformat()} ===")
                print("No matching daily discussion thread found for this date.")
                print()
                continue

            # 3) Try post-id cache (still avoids fetching comments).
            if cache_dir is not None and not refresh_cache:
                cached = load_cached_daily(cache_path_for_post(cache_dir, args.subreddit, chosen.post_id))
                if cached is not None:
                    cached_post, cached_records = cached
                    # Also backfill date-based cache for faster future runs.
                    try:
                        save_cached_daily(
                            cache_path_for_date(cache_dir, args.subreddit, date_value),
                            subreddit=args.subreddit,
                            post=cached_post,
                            records=cached_records,
                        )
                    except OSError:
                        pass
                    print(f"=== {date_value.isoformat()} ===")
                    print(summarize_comment_records(cached_records, cached_post))
                    print()
                    continue

            # 4) Fetch from Reddit and cache.
            comments_url = build_comments_url(
                post_id=chosen.post_id,
                sort=args.comment_sort,
                limit=args.comment_limit,
            )
            try:
                payload = fetch_json(comments_url, args.user_agent, args.timeout)
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1

            if not isinstance(payload, list) or len(payload) < 2:
                print("Error: Unexpected response payload from Reddit.", file=sys.stderr)
                return 1

            post_listing = payload[0]
            comments_listing = payload[1]
            post_from_listing = parse_post_from_listing(post_listing) or chosen

            max_comments = args.max_comments if args.max_comments > 0 else None
            comments = extract_comments(
                payload=comments_listing,
                include_deleted=args.include_deleted,
                min_score=args.min_score,
                max_comments=max_comments,
            )
            records = list(iter_comment_records(comments, post_from_listing))

            if cache_dir is not None:
                try:
                    save_cached_daily(
                        cache_path_for_post(cache_dir, args.subreddit, post_from_listing.post_id),
                        subreddit=args.subreddit,
                        post=post_from_listing,
                        records=records,
                    )
                    save_cached_daily(
                        cache_path_for_date(cache_dir, args.subreddit, date_value),
                        subreddit=args.subreddit,
                        post=post_from_listing,
                        records=records,
                    )
                except OSError:
                    pass

            print(f"=== {date_value.isoformat()} ===")
            print(summarize_comment_records(records, post_from_listing))
            print()

        return 0

    # If the user asked for a specific date, try the date-based cache before searching.
    if cache_dir is not None and not refresh_cache:
        target_date_for_cache: Optional[dt.date] = None
        if args.date is not None:
            target_date_for_cache = args.date
        elif args.days_ago is not None:
            target_date_for_cache = utc_today() - dt.timedelta(days=int(args.days_ago))

        if target_date_for_cache is not None:
            cached = load_cached_daily(cache_path_for_date(cache_dir, args.subreddit, target_date_for_cache))
            if cached is not None:
                cached_post, cached_records = cached

                summary_only = args.summary_only
                summary_text = None
                if args.summary or summary_only:
                    summary_text = summarize_comment_records(cached_records, cached_post)

                if summary_only:
                    print(summary_text)
                else:
                    output_handle = None
                    try:
                        if args.output:
                            output_handle = open(args.output, "w", newline="", encoding="utf-8")
                        else:
                            output_handle = sys.stdout

                        if args.format == "csv":
                            write_csv_records(cached_records, output_handle)
                        elif args.format == "json":
                            pretty = output_handle is sys.stdout and sys.stdout.isatty()
                            write_json_records(cached_records, output_handle, pretty=pretty)
                        else:
                            if output_handle is sys.stdout and sys.stdout.isatty():
                                write_json_records(cached_records, output_handle, pretty=True)
                            else:
                                write_jsonl_records(cached_records, output_handle)
                    finally:
                        if output_handle is not None and output_handle is not sys.stdout:
                            output_handle.close()

                if args.summary and not summary_only and summary_text is not None:
                    print(summary_text, file=sys.stderr)

                print(
                    f"Fetched {len(cached_records)} cached comments from '{cached_post.title}' ({cached_post.url})",
                    file=sys.stderr,
                )
                return 0

    if args.post_id:
        post_id = args.post_id.strip()
    elif args.post_url:
        try:
            post_id = extract_post_id(args.post_url.strip())
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        target_date = None
        if args.date is not None:
            target_date = args.date
        elif args.days_ago is not None:
            target_date = utc_today() - dt.timedelta(days=int(args.days_ago))

        try:
            post = find_daily_post(args, target_date=target_date)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        post_id = post.post_id

    # For post-based requests, try cache after we know the post_id (but before fetching comments).
    if cache_dir is not None and not refresh_cache and post_id is not None:
        cached = load_cached_daily(cache_path_for_post(cache_dir, args.subreddit, post_id))
        if cached is not None:
            cached_post, cached_records = cached

            summary_only = args.summary_only
            summary_text = None
            if args.summary or summary_only:
                summary_text = summarize_comment_records(cached_records, cached_post)

            if summary_only:
                print(summary_text)
            else:
                output_handle = None
                try:
                    if args.output:
                        output_handle = open(args.output, "w", newline="", encoding="utf-8")
                    else:
                        output_handle = sys.stdout

                    if args.format == "csv":
                        write_csv_records(cached_records, output_handle)
                    elif args.format == "json":
                        pretty = output_handle is sys.stdout and sys.stdout.isatty()
                        write_json_records(cached_records, output_handle, pretty=pretty)
                    else:
                        if output_handle is sys.stdout and sys.stdout.isatty():
                            write_json_records(cached_records, output_handle, pretty=True)
                        else:
                            write_jsonl_records(cached_records, output_handle)
                finally:
                    if output_handle is not None and output_handle is not sys.stdout:
                        output_handle.close()

            if args.summary and not summary_only and summary_text is not None:
                print(summary_text, file=sys.stderr)

            print(
                f"Fetched {len(cached_records)} cached comments from '{cached_post.title}' ({cached_post.url})",
                file=sys.stderr,
            )
            return 0

    comments_url = build_comments_url(
        post_id=post_id,
        sort=args.comment_sort,
        limit=args.comment_limit,
    )
    try:
        payload = fetch_json(comments_url, args.user_agent, args.timeout)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not isinstance(payload, list) or len(payload) < 2:
        print("Error: Unexpected response payload from Reddit.", file=sys.stderr)
        return 1

    post_listing = payload[0]
    comments_listing = payload[1]
    if post is None:
        post = parse_post_from_listing(post_listing)

    if post is None:
        print("Error: Unable to parse post metadata.", file=sys.stderr)
        return 1

    max_comments = args.max_comments if args.max_comments > 0 else None
    comments = extract_comments(
        payload=comments_listing,
        include_deleted=args.include_deleted,
        min_score=args.min_score,
        max_comments=max_comments,
    )
    records_for_cache = list(iter_comment_records(comments, post))
    if cache_dir is not None:
        try:
            save_cached_daily(
                cache_path_for_post(cache_dir, args.subreddit, post.post_id),
                subreddit=args.subreddit,
                post=post,
                records=records_for_cache,
            )
            if args.date is not None:
                save_cached_daily(
                    cache_path_for_date(cache_dir, args.subreddit, args.date),
                    subreddit=args.subreddit,
                    post=post,
                    records=records_for_cache,
                )
            elif args.days_ago is not None:
                date_value = utc_today() - dt.timedelta(days=int(args.days_ago))
                save_cached_daily(
                    cache_path_for_date(cache_dir, args.subreddit, date_value),
                    subreddit=args.subreddit,
                    post=post,
                    records=records_for_cache,
                )
        except OSError:
            pass

    summary_only = args.summary_only
    summary_text = None
    if args.summary or summary_only:
        summary_text = summarize_comments(comments, post)

    if summary_only:
        print(summary_text)
    else:
        output_handle = None
        try:
            if args.output:
                output_handle = open(args.output, "w", newline="", encoding="utf-8")
            else:
                output_handle = sys.stdout

            if args.format == "csv":
                write_csv(comments, post, output_handle)
            elif args.format == "json":
                pretty = output_handle is sys.stdout and sys.stdout.isatty()
                write_json(comments, post, output_handle, pretty=pretty)
            else:
                if output_handle is sys.stdout and sys.stdout.isatty():
                    # JSONL is great for files/pipes, but hard to read in a terminal.
                    write_json(comments, post, output_handle, pretty=True)
                else:
                    write_jsonl(comments, post, output_handle)
        finally:
            if output_handle is not None and output_handle is not sys.stdout:
                output_handle.close()

    if args.summary and not summary_only and summary_text is not None:
        print(summary_text, file=sys.stderr)

    print(
        f"Fetched {len(comments)} comments from '{post.title}' ({post.url})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
