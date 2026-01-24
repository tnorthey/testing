#!/usr/bin/env python3
"""
Fetch comments from Reddit, Discord, or Telegram discussions.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional


REDDIT_API_BASE = "https://www.reddit.com"
DISCORD_API_BASE = "https://discord.com/api/v10"
TELEGRAM_API_BASE = "https://api.telegram.org"
OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_QUERIES = "Daily Discussion,Daily General Discussion"
DEFAULT_USER_AGENT = "crypto-price-correlation-script (reddit-daily-comments)"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
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


@dataclass(frozen=True)
class SummaryContext:
    source: str
    title: str
    url: str


@dataclass(frozen=True)
class TextItem:
    author: str
    body: str
    created_utc: int
    score: int


@dataclass(frozen=True)
class DiscordMessage:
    message_id: str
    channel_id: str
    author_id: str
    author_name: str
    content: str
    created_utc: int


@dataclass(frozen=True)
class TelegramMessage:
    message_id: str
    chat_id: str
    author_id: str
    author_name: str
    content: str
    created_utc: int


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


def get_discord_token(value: Optional[str]) -> str:
    token = value or os.getenv("DISCORD_TOKEN", "")
    if not token:
        raise RuntimeError(
            "Discord token is required. Provide --discord-token or set DISCORD_TOKEN."
        )
    return token.strip()


def normalize_discord_auth(token: str) -> str:
    lowered = token.lower()
    if lowered.startswith("bot ") or lowered.startswith("bearer "):
        return token
    return f"Bot {token}"


def build_discord_messages_url(channel_id: str, limit: int, before: Optional[str]) -> str:
    params = {"limit": str(limit)}
    if before:
        params["before"] = before
    return f"{DISCORD_API_BASE}/channels/{channel_id}/messages?{urllib.parse.urlencode(params)}"


def format_discord_author(author: object) -> str:
    if not isinstance(author, dict):
        return "unknown"
    username = str(author.get("username") or "")
    discriminator = str(author.get("discriminator") or "")
    global_name = author.get("global_name")
    if isinstance(global_name, str) and global_name.strip():
        name = global_name.strip()
    else:
        name = username or "unknown"
    if username and discriminator and discriminator not in ("0", "0000"):
        name = f"{username}#{discriminator}"
    return name


def parse_discord_message(data: object, channel_id: str) -> Optional[DiscordMessage]:
    if not isinstance(data, dict):
        return None
    message_id = data.get("id")
    if not message_id:
        return None
    author = data.get("author", {})
    author_id = author.get("id", "unknown") if isinstance(author, dict) else "unknown"
    content = data.get("content") or ""
    timestamp = data.get("timestamp") or ""
    try:
        created_utc = parse_iso_timestamp(str(timestamp))
    except ValueError:
        return None
    return DiscordMessage(
        message_id=str(message_id),
        channel_id=channel_id,
        author_id=str(author_id),
        author_name=format_discord_author(author),
        content=str(content),
        created_utc=created_utc,
    )


def fetch_discord_json(url: str, token: str, timeout: int, max_retries: int = 3) -> object:
    headers = {
        "Authorization": normalize_discord_auth(token),
        "User-Agent": DEFAULT_USER_AGENT,
    }
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            if exc.code == 429 and attempt < max_retries:
                retry_after = 1.0
                if body:
                    try:
                        data = json.loads(body)
                        retry_after = float(data.get("retry_after", retry_after))
                    except (ValueError, TypeError):
                        retry_after = 1.0
                time.sleep(retry_after)
                continue
            message = body.strip() or f"{exc.code} {exc.reason}"
            raise RuntimeError(f"Discord request failed: {message}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Discord request failed: {exc.reason}") from exc
    raise RuntimeError("Discord request failed after retries.")


def fetch_discord_messages(
    channel_id: str,
    token: str,
    limit: int,
    timeout: int,
    before: Optional[str],
) -> list[DiscordMessage]:
    if limit <= 0:
        return []
    messages: list[DiscordMessage] = []
    remaining = limit
    before_id = before
    while remaining > 0:
        batch = min(100, remaining)
        url = build_discord_messages_url(channel_id, batch, before_id)
        payload = fetch_discord_json(url, token, timeout)
        if not isinstance(payload, list):
            raise RuntimeError("Unexpected response payload from Discord.")
        if not payload:
            break
        for item in payload:
            message = parse_discord_message(item, channel_id)
            if message:
                messages.append(message)
        if len(payload) < batch:
            break
        before_id = payload[-1].get("id") if isinstance(payload[-1], dict) else None
        if not before_id:
            break
        remaining = limit - len(messages)
    messages.sort(key=lambda message: message.created_utc)
    return messages


def format_timestamp(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat()


def parse_iso_timestamp(value: str) -> int:
    value = value.strip()
    if not value:
        raise ValueError("Empty timestamp value")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported timestamp format: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp())


def get_telegram_token(value: Optional[str]) -> str:
    token = value or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise RuntimeError(
            "Telegram bot token is required. Provide --telegram-bot-token or set TELEGRAM_BOT_TOKEN."
        )
    return token.strip()


def build_telegram_url(token: str, method: str, params: dict[str, str]) -> str:
    base = f"{TELEGRAM_API_BASE}/bot{token}/{method}"
    if params:
        return f"{base}?{urllib.parse.urlencode(params)}"
    return base


def fetch_telegram_json(url: str, timeout: int) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        message = error_body.strip() or f"{exc.code} {exc.reason}"
        raise RuntimeError(f"Telegram request failed: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Telegram request failed: {exc.reason}") from exc
    return json.loads(payload)


def extract_telegram_result(payload: object) -> object:
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected response payload from Telegram.")
    if not payload.get("ok"):
        description = payload.get("description") or "Telegram API error."
        raise RuntimeError(f"Telegram request failed: {description}")
    return payload.get("result")


def resolve_telegram_chat_id(token: str, chat_id: str, timeout: int) -> str:
    if chat_id.lstrip("-").isdigit():
        return chat_id
    params = {"chat_id": chat_id}
    url = build_telegram_url(token, "getChat", params)
    payload = fetch_telegram_json(url, timeout)
    result = extract_telegram_result(payload)
    if isinstance(result, dict) and "id" in result:
        return str(result["id"])
    raise RuntimeError("Unable to resolve Telegram chat id.")


def format_telegram_author(user: object, sender_chat: object) -> tuple[str, str]:
    if isinstance(user, dict):
        user_id = str(user.get("id") or "")
        username = user.get("username")
        first = user.get("first_name") or ""
        last = user.get("last_name") or ""
        name = " ".join(part for part in (first, last) if part).strip()
        if isinstance(username, str) and username:
            name = f"@{username}"
        if not name:
            name = "unknown"
        return user_id or "unknown", name
    if isinstance(sender_chat, dict):
        chat_id = str(sender_chat.get("id") or "")
        title = sender_chat.get("title") or sender_chat.get("username") or "unknown"
        return chat_id or "unknown", str(title)
    return "unknown", "unknown"


def parse_telegram_message(message: object, target_chat_id: str) -> Optional[TelegramMessage]:
    if not isinstance(message, dict):
        return None
    chat = message.get("chat")
    if not isinstance(chat, dict):
        return None
    chat_id = str(chat.get("id") or "")
    if not chat_id or chat_id != target_chat_id:
        return None
    message_id = message.get("message_id")
    if message_id is None:
        return None
    content = message.get("text") or message.get("caption") or ""
    if not content:
        return None
    created_utc = message.get("date")
    if created_utc is None:
        return None
    author_id, author_name = format_telegram_author(
        message.get("from"), message.get("sender_chat")
    )
    return TelegramMessage(
        message_id=str(message_id),
        chat_id=chat_id,
        author_id=author_id,
        author_name=author_name,
        content=str(content),
        created_utc=int(created_utc),
    )


def extract_telegram_messages(update: object, target_chat_id: str) -> list[TelegramMessage]:
    if not isinstance(update, dict):
        return []
    messages: list[TelegramMessage] = []
    for key in ("message", "edited_message", "channel_post", "edited_channel_post"):
        payload = update.get(key)
        message = parse_telegram_message(payload, target_chat_id)
        if message:
            messages.append(message)
    return messages


def fetch_telegram_updates(
    token: str,
    timeout: int,
    long_poll: int,
    limit: int,
    offset: Optional[int],
) -> list[object]:
    params: dict[str, str] = {"limit": str(limit)}
    if offset is not None:
        params["offset"] = str(offset)
    if long_poll > 0:
        params["timeout"] = str(long_poll)
    url = build_telegram_url(token, "getUpdates", params)
    payload = fetch_telegram_json(url, timeout)
    result = extract_telegram_result(payload)
    if not isinstance(result, list):
        raise RuntimeError("Unexpected result format from Telegram.")
    return result


def fetch_telegram_messages(
    chat_id: str,
    token: str,
    limit: int,
    timeout: int,
    long_poll: int,
    offset: Optional[int],
    max_updates: int,
) -> list[TelegramMessage]:
    if limit <= 0:
        return []
    messages: list[TelegramMessage] = []
    seen_ids: set[str] = set()
    remaining = limit
    next_offset = offset
    updates_seen = 0

    while remaining > 0 and updates_seen < max_updates:
        batch_limit = min(100, remaining)
        updates = fetch_telegram_updates(
            token=token,
            timeout=timeout,
            long_poll=long_poll,
            limit=batch_limit,
            offset=next_offset,
        )
        if not updates:
            break
        updates_seen += len(updates)
        for update in updates:
            for message in extract_telegram_messages(update, chat_id):
                if message.message_id in seen_ids:
                    continue
                seen_ids.add(message.message_id)
                messages.append(message)
                if len(messages) >= limit:
                    break
            if len(messages) >= limit:
                break
        last_update_id = updates[-1].get("update_id") if isinstance(updates[-1], dict) else None
        if last_update_id is None:
            break
        next_offset = int(last_update_id) + 1
        remaining = limit - len(messages)

    messages.sort(key=lambda message: message.created_utc)
    return messages


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


def context_from_reddit_post(post: RedditPost) -> SummaryContext:
    return SummaryContext(source="Reddit", title=post.title, url=post.url)


def build_discord_channel_url(channel_id: str, guild_id: Optional[str]) -> str:
    if guild_id:
        return f"https://discord.com/channels/{guild_id}/{channel_id}"
    return f"discord://channel/{channel_id}"


def build_discord_message_url(
    guild_id: Optional[str], channel_id: str, message_id: str
) -> str:
    if guild_id:
        return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
    return ""


def context_for_discord_channel(channel_id: str, guild_id: Optional[str]) -> SummaryContext:
    title = f"Discord channel {channel_id}"
    if guild_id:
        title = f"Discord channel {channel_id} (guild {guild_id})"
    return SummaryContext(
        source="Discord",
        title=title,
        url=build_discord_channel_url(channel_id, guild_id),
    )


def build_telegram_chat_url(chat_id: str, username: Optional[str]) -> str:
    if username:
        return f"https://t.me/{username}"
    if chat_id.startswith("-100"):
        internal_id = chat_id[4:]
        if internal_id:
            return f"https://t.me/c/{internal_id}"
    return ""


def build_telegram_message_url(chat_id: str, message_id: str, username: Optional[str]) -> str:
    if username:
        return f"https://t.me/{username}/{message_id}"
    if chat_id.startswith("-100"):
        internal_id = chat_id[4:]
        if internal_id:
            return f"https://t.me/c/{internal_id}/{message_id}"
    return ""


def context_for_telegram_chat(chat_id: str, username: Optional[str]) -> SummaryContext:
    title = f"Telegram chat {chat_id}"
    if username:
        title = f"Telegram chat @{username}"
    return SummaryContext(
        source="Telegram",
        title=title,
        url=build_telegram_chat_url(chat_id, username),
    )


def reddit_comments_to_items(comments: Iterable[RedditComment]) -> list[TextItem]:
    return [
        TextItem(
            author=comment.author,
            body=comment.body,
            created_utc=comment.created_utc,
            score=comment.score,
        )
        for comment in comments
    ]


def discord_messages_to_items(messages: Iterable[DiscordMessage]) -> list[TextItem]:
    return [
        TextItem(
            author=message.author_name,
            body=message.content,
            created_utc=message.created_utc,
            score=0,
        )
        for message in messages
    ]


def telegram_messages_to_items(messages: Iterable[TelegramMessage]) -> list[TextItem]:
    return [
        TextItem(
            author=message.author_name,
            body=message.content,
            created_utc=message.created_utc,
            score=0,
        )
        for message in messages
    ]


def summarize_text_items(
    items: Iterable[TextItem],
    context: SummaryContext,
    max_terms: int = 10,
    max_authors: int = 5,
    max_examples: int = 3,
) -> str:
    items_list = list(items)
    if not items_list:
        return "No comments available to summarize."

    author_counts = Counter(item.author for item in items_list)
    word_counts: Counter[str] = Counter()
    ticker_counts: Counter[str] = Counter()
    timestamps = [item.created_utc for item in items_list]

    for item in items_list:
        body = item.body
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
    top_scored = sorted(
        items_list,
        key=lambda item: (item.score, item.created_utc),
        reverse=True,
    )[:max_examples]
    has_scores = any(item.score for item in items_list)

    lines = [
        f"Summary for '{to_ascii(context.title)}'",
        f"Source: {context.source}",
        f"URL: {context.url or 'N/A'}",
        f"Entries: {len(items_list)} | Unique authors: {len(author_counts)}",
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
        if has_scores:
            lines.append("Top scored comments:")
            for item in top_scored:
                snippet = truncate_text(item.body, 160)
                lines.append(f"- ({item.score}) {snippet}")
        else:
            lines.append("Sample messages:")
            for item in top_scored:
                snippet = truncate_text(item.body, 160)
                lines.append(f"- {snippet}")

    return "\n".join(lines)


def build_comment_corpus(
    comments: Iterable[TextItem],
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


def build_ai_prompt(
    context: SummaryContext,
    total_comments: int,
    sample_comments: int,
    corpus: str,
) -> str:
    header = [
        f"You are summarizing messages from a {context.source} discussion.",
        f"Title: {to_ascii(context.title)}",
        f"URL: {context.url or 'N/A'}",
        f"Total entries fetched: {total_comments}",
        f"Sample entries provided: {sample_comments}",
        "",
        "Instructions:",
        "- Provide 4-6 concise bullet themes.",
        "- State overall sentiment (bullish, bearish, mixed).",
        "- Mention any repeated tickers or projects.",
        "- Mention notable questions or debates (1-2).",
        "- Keep under 200 words.",
        "",
        "Entries (each line is '(score) message'):",
    ]
    if not corpus:
        corpus = "- (0) No message text available."
    return "\n".join(header) + "\n" + corpus


def get_openai_api_key(value: Optional[str]) -> str:
    api_key = value or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key is required. Provide --openai-api-key or set OPENAI_API_KEY."
        )
    return api_key


def run_openai_chat(
    prompt: str,
    model: str,
    api_key: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You summarize discussion threads with concise, factual bullets.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OPENAI_API_BASE}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        message = error_body.strip() or f"{exc.code} {exc.reason}"
        raise RuntimeError(f"OpenAI request failed: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc

    parsed = json.loads(response_data)
    choices = parsed.get("choices", [])
    if not choices:
        raise RuntimeError("OpenAI response missing choices.")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("OpenAI response missing content.")
    return content.strip()


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
    items: Iterable[TextItem],
    context: SummaryContext,
    model: str,
    max_chars: int,
    max_comments: int,
    timeout: int,
) -> str:
    items_list = list(items)
    if not items_list:
        return "No comments available to summarize."
    sorted_items = sorted(
        items_list, key=lambda item: (item.score, item.created_utc), reverse=True
    )
    corpus, sample_count = build_comment_corpus(sorted_items, max_chars, max_comments)
    prompt = build_ai_prompt(context, len(items_list), sample_count, corpus)
    return run_ollama(prompt, model, timeout)


def summarize_with_chatgpt(
    items: Iterable[TextItem],
    context: SummaryContext,
    model: str,
    api_key: str,
    max_chars: int,
    max_comments: int,
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> str:
    items_list = list(items)
    if not items_list:
        return "No comments available to summarize."
    sorted_items = sorted(
        items_list, key=lambda item: (item.score, item.created_utc), reverse=True
    )
    corpus, sample_count = build_comment_corpus(sorted_items, max_chars, max_comments)
    prompt = build_ai_prompt(context, len(items_list), sample_count, corpus)
    return run_openai_chat(prompt, model, api_key, timeout, max_tokens, temperature)


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


def write_discord_jsonl(
    messages: Iterable[DiscordMessage],
    channel_id: str,
    guild_id: Optional[str],
    handle: object,
) -> None:
    channel_url = build_discord_channel_url(channel_id, guild_id)
    for message in messages:
        record = {
            "source": "discord",
            "channel_id": message.channel_id,
            "channel_url": channel_url,
            "message_id": message.message_id,
            "message_url": build_discord_message_url(
                guild_id, message.channel_id, message.message_id
            ),
            "author_id": message.author_id,
            "author_name": message.author_name,
            "content": message.content,
            "created_utc": message.created_utc,
            "created_iso": format_timestamp(message.created_utc),
        }
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_discord_csv(
    messages: Iterable[DiscordMessage],
    channel_id: str,
    guild_id: Optional[str],
    handle: object,
) -> None:
    channel_url = build_discord_channel_url(channel_id, guild_id)
    writer = csv.writer(handle)
    writer.writerow(
        [
            "source",
            "channel_id",
            "channel_url",
            "message_id",
            "message_url",
            "author_id",
            "author_name",
            "content",
            "created_utc",
            "created_iso",
        ]
    )
    for message in messages:
        writer.writerow(
            [
                "discord",
                message.channel_id,
                channel_url,
                message.message_id,
                build_discord_message_url(guild_id, message.channel_id, message.message_id),
                message.author_id,
                message.author_name,
                message.content,
                message.created_utc,
                format_timestamp(message.created_utc),
            ]
        )


def write_telegram_jsonl(
    messages: Iterable[TelegramMessage],
    chat_id: str,
    chat_username: Optional[str],
    handle: object,
) -> None:
    chat_url = build_telegram_chat_url(chat_id, chat_username)
    for message in messages:
        record = {
            "source": "telegram",
            "chat_id": message.chat_id,
            "chat_url": chat_url,
            "message_id": message.message_id,
            "message_url": build_telegram_message_url(
                message.chat_id, message.message_id, chat_username
            ),
            "author_id": message.author_id,
            "author_name": message.author_name,
            "content": message.content,
            "created_utc": message.created_utc,
            "created_iso": format_timestamp(message.created_utc),
        }
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_telegram_csv(
    messages: Iterable[TelegramMessage],
    chat_id: str,
    chat_username: Optional[str],
    handle: object,
) -> None:
    chat_url = build_telegram_chat_url(chat_id, chat_username)
    writer = csv.writer(handle)
    writer.writerow(
        [
            "source",
            "chat_id",
            "chat_url",
            "message_id",
            "message_url",
            "author_id",
            "author_name",
            "content",
            "created_utc",
            "created_iso",
        ]
    )
    for message in messages:
        writer.writerow(
            [
                "telegram",
                message.chat_id,
                chat_url,
                message.message_id,
                build_telegram_message_url(
                    message.chat_id, message.message_id, chat_username
                ),
                message.author_id,
                message.author_name,
                message.content,
                message.created_utc,
                format_timestamp(message.created_utc),
            ]
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Reddit, Discord, or Telegram discussion comments."
    )
    parser.add_argument(
        "--source",
        default="reddit",
        choices=("reddit", "discord", "telegram"),
        help="Data source to fetch (default: reddit).",
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
        "--discord-token",
        help="Discord bot token (defaults to DISCORD_TOKEN).",
    )
    parser.add_argument(
        "--discord-channel-id",
        help="Discord channel id to fetch messages from.",
    )
    parser.add_argument(
        "--discord-guild-id",
        help="Discord guild id (optional, used to build message URLs).",
    )
    parser.add_argument(
        "--discord-limit",
        type=int,
        default=200,
        help="Max Discord messages to fetch (default: 200).",
    )
    parser.add_argument(
        "--discord-before",
        help="Fetch Discord messages before this message id.",
    )
    parser.add_argument(
        "--telegram-bot-token",
        help="Telegram bot token (defaults to TELEGRAM_BOT_TOKEN).",
    )
    parser.add_argument(
        "--telegram-chat-id",
        help="Telegram chat id to fetch messages from.",
    )
    parser.add_argument(
        "--telegram-chat-username",
        help="Telegram chat username (used to build message URLs).",
    )
    parser.add_argument(
        "--telegram-limit",
        type=int,
        default=200,
        help="Max Telegram messages to fetch (default: 200).",
    )
    parser.add_argument(
        "--telegram-offset",
        type=int,
        help="Telegram update offset to start fetching from.",
    )
    parser.add_argument(
        "--telegram-long-poll",
        type=int,
        default=0,
        help="Telegram long poll duration in seconds (default: 0).",
    )
    parser.add_argument(
        "--telegram-max-updates",
        type=int,
        default=1000,
        help="Max Telegram updates to scan (default: 1000).",
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
        "--summary-chatgpt",
        action="store_true",
        help="Use the ChatGPT API for the summary.",
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
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model name (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (defaults to OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--openai-max-chars",
        type=int,
        default=12000,
        help="Max characters of comment text sent to OpenAI (default: 12000).",
    )
    parser.add_argument(
        "--openai-max-comments",
        type=int,
        default=200,
        help="Max comments sent to OpenAI (default: 200).",
    )
    parser.add_argument(
        "--openai-timeout",
        type=int,
        default=60,
        help="Timeout for OpenAI request in seconds (default: 60).",
    )
    parser.add_argument(
        "--openai-max-tokens",
        type=int,
        default=300,
        help="Max tokens for the OpenAI response (default: 300).",
    )
    parser.add_argument(
        "--openai-temperature",
        type=float,
        default=0.2,
        help="OpenAI temperature setting (default: 0.2).",
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

    if args.summary_ollama and args.summary_chatgpt:
        print("Error: choose only one of --summary-ollama or --summary-chatgpt.", file=sys.stderr)
        return 1
    output_kind = args.source
    output_label = ""
    context: SummaryContext
    items: list[TextItem]

    if args.source == "discord":
        if not args.discord_channel_id:
            print("Error: --discord-channel-id is required for Discord source.", file=sys.stderr)
            return 1
        channel_id = args.discord_channel_id.strip()
        guild_id = args.discord_guild_id.strip() if args.discord_guild_id else None
        try:
            token = get_discord_token(args.discord_token)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        try:
            messages = fetch_discord_messages(
                channel_id=channel_id,
                token=token,
                limit=args.discord_limit,
                timeout=args.timeout,
                before=args.discord_before,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        context = context_for_discord_channel(channel_id, guild_id)
        items = discord_messages_to_items(messages)
        output_label = (
            f"Fetched {len(messages)} messages from Discord channel "
            f"{channel_id}"
        )
    elif args.source == "telegram":
        if not args.telegram_chat_id:
            print("Error: --telegram-chat-id is required for Telegram source.", file=sys.stderr)
            return 1
        raw_chat_id = str(args.telegram_chat_id).strip()
        chat_username = args.telegram_chat_username.strip() if args.telegram_chat_username else None
        if raw_chat_id.startswith("@") and not chat_username:
            chat_username = raw_chat_id.lstrip("@")
        try:
            token = get_telegram_token(args.telegram_bot_token)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        try:
            chat_id = resolve_telegram_chat_id(token, raw_chat_id, args.timeout)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        try:
            telegram_messages = fetch_telegram_messages(
                chat_id=chat_id,
                token=token,
                limit=args.telegram_limit,
                timeout=args.timeout,
                long_poll=args.telegram_long_poll,
                offset=args.telegram_offset,
                max_updates=args.telegram_max_updates,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        context = context_for_telegram_chat(chat_id, chat_username)
        items = telegram_messages_to_items(telegram_messages)
        output_label = (
            f"Fetched {len(telegram_messages)} messages from Telegram chat {chat_id}"
        )
    else:
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
        context = context_from_reddit_post(post)
        items = reddit_comments_to_items(comments)
        output_label = f"Fetched {len(comments)} comments from '{post.title}' ({post.url})"

    summary_only = args.summary_only
    summary_requested = (
        args.summary or summary_only or args.summary_ollama or args.summary_chatgpt
    )
    summary_text = None
    if summary_requested:
        if args.summary_chatgpt:
            try:
                api_key = get_openai_api_key(args.openai_api_key)
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            try:
                summary_text = summarize_with_chatgpt(
                    items=items,
                    context=context,
                    model=args.openai_model,
                    api_key=api_key,
                    max_chars=args.openai_max_chars,
                    max_comments=args.openai_max_comments,
                    timeout=args.openai_timeout,
                    max_tokens=args.openai_max_tokens,
                    temperature=args.openai_temperature,
                )
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
        elif args.summary_ollama:
            try:
                summary_text = summarize_with_ollama(
                    items=items,
                    context=context,
                    model=args.ollama_model,
                    max_chars=args.ollama_max_chars,
                    max_comments=args.ollama_max_comments,
                    timeout=args.ollama_timeout,
                )
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
        else:
            summary_text = summarize_text_items(items, context)

    if summary_only:
        print(summary_text)
    else:
        output_handle = None
        try:
            if args.output:
                output_handle = open(args.output, "w", newline="", encoding="utf-8")
            else:
                output_handle = sys.stdout

            if output_kind == "discord":
                if args.format == "csv":
                    write_discord_csv(
                        messages=messages,
                        channel_id=channel_id,
                        guild_id=guild_id,
                        handle=output_handle,
                    )
                else:
                    write_discord_jsonl(
                        messages=messages,
                        channel_id=channel_id,
                        guild_id=guild_id,
                        handle=output_handle,
                    )
            elif output_kind == "telegram":
                if args.format == "csv":
                    write_telegram_csv(
                        messages=telegram_messages,
                        chat_id=chat_id,
                        chat_username=chat_username,
                        handle=output_handle,
                    )
                else:
                    write_telegram_jsonl(
                        messages=telegram_messages,
                        chat_id=chat_id,
                        chat_username=chat_username,
                        handle=output_handle,
                    )
            else:
                if args.format == "csv":
                    write_csv(comments, post, output_handle)
                else:
                    write_jsonl(comments, post, output_handle)
        finally:
            if output_handle is not None and output_handle is not sys.stdout:
                output_handle.close()

    if summary_requested and not summary_only and summary_text is not None:
        print(summary_text, file=sys.stderr)

    print(output_label, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
