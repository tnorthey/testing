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
from dataclasses import dataclass
from typing import Iterable, Optional


REDDIT_API_BASE = "https://www.reddit.com"
DEFAULT_QUERIES = "Daily Discussion,Daily General Discussion"
DEFAULT_USER_AGENT = "crypto-price-correlation-script (reddit-daily-comments)"
POST_ID_RE = re.compile(r"/comments/([a-z0-9]+)/")


@dataclass(frozen=True)
class RedditPost:
    post_id: str
    title: str
    created_utc: int
    permalink: str
    num_comments: int

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
) -> str:
    params = {
        "q": query,
        "restrict_sr": "1",
        "sort": sort,
        "t": time_filter,
        "limit": str(limit),
        "raw_json": "1",
    }
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
        if not all([post_id, title, created_utc, permalink, num_comments is not None]):
            continue
        posts.append(
            RedditPost(
                post_id=str(post_id),
                title=str(title),
                created_utc=int(created_utc),
                permalink=str(permalink),
                num_comments=int(num_comments),
            )
        )
    return posts


def title_matches_queries(title: str, queries: Iterable[str]) -> bool:
    title_lower = title.lower()
    return any(query.lower() in title_lower for query in queries)


def pick_latest_post(posts: Iterable[RedditPost], queries: list[str]) -> Optional[RedditPost]:
    posts_list = list(posts)
    if not posts_list:
        return None
    matches = [post for post in posts_list if title_matches_queries(post.title, queries)]
    if not matches:
        return max(posts_list, key=lambda post: post.created_utc)
    return max(matches, key=lambda post: post.created_utc)


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
        if not all([post_id, title, created_utc, permalink, num_comments is not None]):
            continue
        return RedditPost(
            post_id=str(post_id),
            title=str(title),
            created_utc=int(created_utc),
            permalink=str(permalink),
            num_comments=int(num_comments),
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


def write_jsonl(
    comments: Iterable[RedditComment],
    post: RedditPost,
    handle: object,
) -> None:
    for comment in comments:
        record = {
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
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


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
    parser.add_argument(
        "--post-id",
        help="Explicit Reddit post id (skips search).",
    )
    parser.add_argument(
        "--post-url",
        help="Explicit Reddit post URL (skips search).",
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
        choices=("jsonl", "csv"),
        help="Output format (default: jsonl).",
    )
    parser.add_argument(
        "--output",
        help="Output path (default: stdout).",
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


def find_daily_post(args: argparse.Namespace) -> RedditPost:
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

    latest = pick_latest_post(posts.values(), queries)
    if latest:
        return latest

    raise RuntimeError("No matching daily discussion thread found.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    post_id = None
    post = None
    if args.post_id:
        post_id = args.post_id.strip()
    elif args.post_url:
        try:
            post_id = extract_post_id(args.post_url.strip())
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        try:
            post = find_daily_post(args)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        post_id = post.post_id

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

    output_handle = None
    try:
        if args.output:
            output_handle = open(args.output, "w", newline="", encoding="utf-8")
        else:
            output_handle = sys.stdout

        if args.format == "csv":
            write_csv(comments, post, output_handle)
        else:
            write_jsonl(comments, post, output_handle)
    finally:
        if output_handle is not None and output_handle is not sys.stdout:
            output_handle.close()

    print(
        f"Fetched {len(comments)} comments from '{post.title}' ({post.url})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
