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
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional


REDDIT_API_BASE = "https://www.reddit.com"
DEFAULT_QUERIES = "Daily Discussion,Daily General Discussion"
DEFAULT_USER_AGENT = "crypto-price-correlation-script (reddit-daily-comments)"
POST_ID_RE = re.compile(r"/comments/([a-z0-9]+)/")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z']{2,}")
TICKER_RE = re.compile(r"\$[A-Za-z]{2,6}")
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


def build_comment_corpus(
    comments: Iterable[RedditComment],
    max_chars: int,
    max_comments: int,
) -> tuple[str, int]:
    lines: list[str] = []
    count = 0
    used = 0
    limit_chars = max_chars if max_chars > 0 else None
    limit_comments = max_comments if max_comments > 0 else None

    for comment in comments:
        if limit_comments and count >= limit_comments:
            break
        body = " ".join(comment.body.split())
        if not body:
            continue
        line = f"- ({comment.score}) {body}"
        if limit_chars is not None and used + len(line) + 1 > limit_chars:
            break
        lines.append(line)
        used += len(line) + 1
        count += 1

    return "\n".join(lines), count


def build_ollama_prompt(
    post: RedditPost,
    total_comments: int,
    sample_comments: int,
    corpus: str,
) -> str:
    header = [
        "You are summarizing comments from a Reddit daily discussion thread.",
        f"Thread title: {to_ascii(post.title)}",
        f"Thread url: {post.url}",
        f"Total comments fetched: {total_comments}",
        f"Sample comments provided: {sample_comments}",
        "",
        "Instructions:",
        "- Provide 4-6 concise bullet themes.",
        "- State overall sentiment (bullish, bearish, mixed).",
        "- Mention any repeated tickers or projects.",
        "- Mention notable questions or debates (1-2).",
        "- Keep under 200 words.",
        "",
        "Comments (each line is '(score) comment'):",
    ]
    if not corpus:
        corpus = "- (0) No comment text available."
    return "\n".join(header) + "\n" + corpus


def run_ollama(prompt: str, model: str, timeout: int) -> str:
    if shutil.which("ollama") is None:
        raise RuntimeError("Ollama CLI not found. Install Ollama to use --summary-ollama.")
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Ollama failed: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Ollama timed out after {timeout} seconds.") from exc

    output = result.stdout.strip()
    if not output:
        raise RuntimeError("Ollama returned an empty response.")
    return output


def summarize_with_ollama(
    comments: Iterable[RedditComment],
    post: RedditPost,
    model: str,
    max_chars: int,
    max_comments: int,
    timeout: int,
) -> str:
    comments_list = list(comments)
    if not comments_list:
        return "No comments available to summarize."
    sorted_comments = sorted(comments_list, key=lambda comment: comment.score, reverse=True)
    corpus, sample_count = build_comment_corpus(sorted_comments, max_chars, max_comments)
    prompt = build_ollama_prompt(post, len(comments_list), sample_count, corpus)
    return run_ollama(prompt, model, timeout)


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
        "--summary-ollama",
        action="store_true",
        help="Use a local Ollama model for the summary.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3",
        help="Ollama model name (default: llama3).",
    )
    parser.add_argument(
        "--ollama-max-chars",
        type=int,
        default=12000,
        help="Max characters of comment text sent to Ollama (default: 12000).",
    )
    parser.add_argument(
        "--ollama-max-comments",
        type=int,
        default=200,
        help="Max comments sent to Ollama (default: 200).",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=180,
        help="Timeout for Ollama run in seconds (default: 180).",
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

    summary_only = args.summary_only
    summary_requested = args.summary or summary_only or args.summary_ollama
    summary_text = None
    if summary_requested:
        if args.summary_ollama:
            summary_text = summarize_with_ollama(
                comments=comments,
                post=post,
                model=args.ollama_model,
                max_chars=args.ollama_max_chars,
                max_comments=args.ollama_max_comments,
                timeout=args.ollama_timeout,
            )
        else:
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
            else:
                write_jsonl(comments, post, output_handle)
        finally:
            if output_handle is not None and output_handle is not sys.stdout:
                output_handle.close()

    if summary_requested and not summary_only and summary_text is not None:
        print(summary_text, file=sys.stderr)

    print(
        f"Fetched {len(comments)} comments from '{post.title}' ({post.url})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
