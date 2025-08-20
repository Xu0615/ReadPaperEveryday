#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arxiv_agent.py  —  异步并发版（采用 AsyncOpenAI，总结模型 gpt-5-mini）

新增特性：
1) 当日文件命名：<SAFE_KEYWORD>_<YYYYMMDD>.md/json
2) 增量更新：
   - 仅对“未收录”的论文做摘要；
   - 无新增：若当日文件已存在 -> 不覆盖；若不存在 -> 写入“没有最新的论文发表”；
   - 有新增：与当日已有内容合并去重，并按提交时间倒序重写当日文件（包含旧+新）。
3) 每个关键词的 seen 集合保存在 state/seen_<SAFE_KEYWORD>.json

环境变量（运行时解析优先级）：
- API Key:  --api-key > ARXIV_AGENT_API_KEY > OPENAI_API_KEY
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import re
import sys
import time
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode, urljoin

import requests
import httpx
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from openai import AsyncOpenAI  # 异步客户端

# ---------------- Configuration ----------------
ARXIV_SEARCH_BASE = "https://arxiv.org/search/cs"
PAGE_SIZE_DEFAULT = 50
REQUEST_TIMEOUT = 30
USER_AGENT = "arxiv-agent/1.1 (+https://github.com/yourname; respectful bot for personal research)"
OPENAI_BASE_URL_DEFAULT = "" # 记得自己设定

MODEL_NAME = "gpt-5-mini"
SYSTEM_PROMPT = (
    "You are a senior AI researcher. Please explain the main work, contributions and "
    "innovations of this paper based on its title and abstract. Please output in Chinese."
)

# --------------- Data structures ---------------
@dataclasses.dataclass
class Paper:
    url: str
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    submitted: datetime


# ---------------- Utilities ----------------
def _requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.7",
        "Connection": "close",
    })
    return s


def build_search_url(keyword: str, start: int = 0, size: int = PAGE_SIZE_DEFAULT) -> str:
    params = {
        "query": keyword,
        "searchtype": "all",
        "abstracts": "show",
        "order": "-announced_date_first",
        "size": str(size),
        "start": str(start),
    }
    return f"{ARXIV_SEARCH_BASE}?{urlencode(params)}"


def parse_submitted_date_from_result(li: BeautifulSoup) -> Optional[datetime]:
    p = li.find("p", class_="is-size-7")
    if not p:
        return None
    text = " ".join(p.stripped_strings)
    m = re.search(r"Submitted\s+([^.;]+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return dateparser.parse(m.group(1).strip(), dayfirst=True, fuzzy=True)
    except Exception:
        return None


def parse_link_from_result(li: BeautifulSoup) -> Optional[str]:
    title_p = li.find("p", class_=re.compile(r"\btitle\b"))
    if title_p and title_p.a and title_p.a.get("href"):
        return urljoin("https://arxiv.org", title_p.a["href"])
    a = li.find("a", href=re.compile(r"/abs/"))
    if a and a.get("href"):
        return urljoin("https://arxiv.org", a["href"])
    return None


def extract_arxiv_id_from_url(url: str) -> str:
    m = re.search(r"/abs/([0-9]+\.[0-9]+)(v[0-9]+)?", url)
    return m.group(1) if m else url


def fetch_html(session: requests.Session, url: str, sleep: float = 1.0, max_retries: int = 6) -> str:
    """抓取页面，带指数退避重试（处理 5xx/429/超时）"""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code >= 500 or r.status_code in (429,):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            if sleep > 0:
                time.sleep(sleep)
            return r.text
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_err = e
            backoff = max(1.0, sleep) * (2 ** attempt) + random.random() * 0.5
            print(f"[WARN] fetch_html retry {attempt+1}/{max_retries} after error: {e}. Backoff {backoff:.2f}s", flush=True)
            time.sleep(backoff)
        except requests.RequestException as e:
            last_err = e
            break
    if last_err:
        raise last_err
    raise RuntimeError("fetch_html failed unexpectedly")


def parse_paper_page(html: str, url: str) -> Paper:
    soup = BeautifulSoup(html, "html.parser")
    # 标题
    h1 = soup.find("h1", class_=re.compile(r"\btitle\b"))
    title = ""
    if h1:
        title = " ".join(h1.stripped_strings)
        title = re.sub(r"^\s*Title:\s*", "", title, flags=re.IGNORECASE)
    # 作者
    authors_div = soup.find("div", class_="authors")
    authors: List[str] = [a.get_text(strip=True) for a in authors_div.find_all("a")] if authors_div else []
    # 摘要
    abstract_block = soup.find("blockquote", class_=re.compile(r"\babstract\b"))
    abstract = ""
    if abstract_block:
        abstract = " ".join(abstract_block.stripped_strings)
        abstract = re.sub(r"^\s*Abstract:\s*", "", abstract, flags=re.IGNORECASE)
    # 时间
    dateline = soup.find("div", class_="dateline")
    submitted_dt = None
    if dateline:
        datetext = " ".join(dateline.stripped_strings)
        m = re.search(r"Submitted\s+(?:on\s+)?([A-Za-z0-9 ,]+)", datetext, flags=re.IGNORECASE)
        date_str = m.group(1) if m else datetext
        try:
            submitted_dt = dateparser.parse(date_str, fuzzy=True)
        except Exception:
            submitted_dt = None
    if not submitted_dt:
        submitted_dt = datetime.now(tz=timezone.utc)
    return Paper(url=url, arxiv_id=extract_arxiv_id_from_url(url), title=title,
                 authors=authors, abstract=abstract, submitted=submitted_dt)


# ---------- 增量记录：seen 集合 ----------
def _safe_kw(keyword: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", keyword)[:50]


def _seen_file(state_dir: str, safe_kw: str) -> str:
    return os.path.join(state_dir, f"seen_{safe_kw}.json")


def _load_seen(state_dir: str, safe_kw: str) -> Set[str]:
    path = _seen_file(state_dir, safe_kw)
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = set(data.get("arxiv_ids", []))
        return ids
    except Exception:
        return set()


def _save_seen(state_dir: str, safe_kw: str, ids: Set[str]) -> None:
    os.makedirs(state_dir, exist_ok=True)
    path = _seen_file(state_dir, safe_kw)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"arxiv_ids": sorted(ids)}, f, ensure_ascii=False, indent=2)


def _derive_date_stamp(outdir: str) -> str:
    """优先从 outdir 末尾抽取 YYYYMMDD；否则用本地日期。"""
    m = re.search(r"(\d{8})/?$", outdir.strip())
    if m:
        return m.group(1)
    return datetime.now().strftime("%Y%m%d")


def _load_existing_report(json_path: str) -> Tuple[List[Paper], Dict[str, Optional[str]]]:
    """
    从已存在的当日 JSON 恢复 Paper 列表与每篇的 model_summary
    """
    papers: List[Paper] = []
    summaries: Dict[str, Optional[str]] = {}
    if not os.path.exists(json_path):
        return papers, summaries
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for it in data.get("papers", []):
            submitted_str = it.get("submitted") or ""
            try:
                submitted_dt = dateparser.parse(submitted_str).replace(tzinfo=None)
            except Exception:
                submitted_dt = datetime.now()
            p = Paper(
                url=it.get("url", ""),
                arxiv_id=it.get("arxiv_id", ""),
                title=it.get("title", ""),
                authors=it.get("authors", []),
                abstract=it.get("abstract", ""),
                submitted=submitted_dt,
            )
            papers.append(p)
            summaries[p.url] = it.get("model_summary")
    except Exception:
        pass
    return papers, summaries


# ---------------- Async OpenAI summarizer ----------------
@dataclasses.dataclass
class SummarizerConfig:
    model: str = MODEL_NAME
    timeout: float = 60.0
    base_url: Optional[str] = None   # 优先级：OPENAI_BASE_URL > ARXIV_AGENT_BASE_URL > base_url > 默认
    max_retries: int = 3
    concurrent: int = 4
    temperature: Optional[float] = 0.2  # 模型不支持时自动去掉
    debug: bool = False
    debug_truncate: int = 800

class AsyncOpenAISummarizer:
    def __init__(self, api_key: str, cfg: SummarizerConfig):
        self.api_key = api_key
        self.cfg = cfg
        self.base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("ARXIV_AGENT_BASE_URL")
            or cfg.base_url
            or OPENAI_BASE_URL_DEFAULT
        ).rstrip("/")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=cfg.timeout,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
                headers={"Connection": "close"},
            ),
            max_retries=cfg.max_retries,
        )

    async def _sdk_chat(self, messages: List[Dict], temperature: Optional[float]) -> str:
        kwargs = {
            "model": self.cfg.model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = await self.client.chat.completions.create(**kwargs)
        content = (resp.choices[0].message.content or "").strip()
        return content if content else "[Empty model response]"

    async def _raw_chat(self, messages: List[Dict], param_name: str, value: int,
                        temperature: Optional[float]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            param_name: value,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.cfg.timeout) as http:
            r = await http.post(url, json=payload, headers=headers)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text}") from e
            data = r.json()
            try:
                content = (data["choices"][0]["message"]["content"] or "").strip()
            except Exception:
                content = ""
            return content if content else "[Empty model response]"

    async def ask_once(self, title: str, abstract: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"},
        ]
        # 1) 先走 SDK
        try:
            return await self._sdk_chat(messages, temperature=self.cfg.temperature)
        except Exception as e:
            msg = str(e)
            if "not_authorized_invalid_key_type" in msg or "does not allow user keys" in msg:
                raise RuntimeError(
                    "Your key is not permitted for this project/model. "
                    "Please use a Project key with access to gpt-5, or enable user keys for this project."
                ) from e
            # 温度不支持 → 去掉温度
            if ("Unsupported value" in msg and "'temperature'" in msg):
                try:
                    return await self._sdk_chat(messages, temperature=None)
                except Exception as e2:
                    msg2 = str(e2)
                    if ("Unsupported parameter" in msg2 and "'max_tokens'" in msg2):
                        try:
                            return await self._raw_chat(messages, "max_completion_tokens", 600, temperature=None)
                        except Exception as e3:
                            try:
                                return await self._raw_chat(messages, "max_output_tokens", 600, temperature=None)
                            except Exception as e4:
                                raise RuntimeError(f"Fallback failed: {e3} / {e4}") from e4
                    try:
                        return await self._raw_chat(messages, "max_completion_tokens", 600, temperature=None)
                    except Exception:
                        return await self._raw_chat(messages, "max_output_tokens", 600, temperature=None)
            # 非温度问题：可能是 max_tokens 不支持 → HTTP 回退
            if "Unsupported parameter" in msg and "'max_tokens'" in msg:
                try:
                    return await self._raw_chat(messages, "max_completion_tokens", 600, temperature=self.cfg.temperature)
                except Exception as e2:
                    try:
                        return await self._raw_chat(messages, "max_output_tokens", 600, temperature=self.cfg.temperature)
                    except Exception as e3:
                        raise RuntimeError(f"Fallback failed: {e2} / {e3}") from e3
            # 兜底两次 HTTP 回退
            try:
                return await self._raw_chat(messages, "max_completion_tokens", 600, temperature=self.cfg.temperature)
            except Exception:
                return await self._raw_chat(messages, "max_output_tokens", 600, temperature=self.cfg.temperature)

    async def summarize_many(self, papers: List["Paper"], sleep_between: float = 0.0) -> Dict[str, str]:
        summaries: Dict[str, str] = {}
        sem = asyncio.Semaphore(max(1, self.cfg.concurrent))

        async def worker(p: "Paper", idx: int, total: int):
            async with sem:
                print(f"[INFO] 模型解读 {idx}/{total}：{p.title[:60]}...", flush=True)
                try:
                    out = await self.ask_once(p.title, p.abstract)
                except Exception as e:
                    out = f"[Summarization error: {e}]"
                summaries[p.url] = out
                if sleep_between and sleep_between > 0:
                    await asyncio.sleep(sleep_between)

        await asyncio.gather(*(worker(p, i + 1, len(papers)) for i, p in enumerate(papers)))
        return summaries


# ---------------- Core search workflow ----------------
def search_keyword(keyword: str, since: datetime, page_size: int = PAGE_SIZE_DEFAULT,
                   max_pages: int = 20, polite_sleep: float = 1.0,
                   session: Optional[requests.Session] = None) -> List[str]:
    s = session or _requests_session()
    links: List[str] = []
    for page_idx in range(max_pages):
        start = page_idx * page_size
        url = build_search_url(keyword, start=start, size=page_size)
        print(f"[INFO] 列表页 {page_idx+1} start={start}  URL={url}", flush=True)
        html = fetch_html(s, url, sleep=polite_sleep)
        soup = BeautifulSoup(html, "html.parser")
        results = soup.find_all("li", class_="arxiv-result") or soup.select("li.arxiv-result, li[class*=arxiv-result], li[itemscope]")
        older_hit = False
        page_links: List[str] = []
        for li in results:
            sub_dt = parse_submitted_date_from_result(li)
            link = parse_link_from_result(li)
            if not link:
                continue
            if sub_dt and sub_dt < since:
                older_hit = True
                break
            page_links.append(link)
        links.extend(page_links)
        if older_hit:
            print("[INFO] 本页出现早于 since 的论文，按规则停止继续翻页。", flush=True)
            break
        if len(results) < page_size:
            print("[INFO] 结果少于 page_size，已到最后一页。", flush=True)
            break
    # 去重保序
    seen = set()
    unique = []
    for u in links:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def collect_papers(links: List[str], polite_sleep: float = 1.0,
                   session: Optional[requests.Session] = None) -> List[Paper]:
    s = session or _requests_session()
    papers: List[Paper] = []
    total = len(links)
    for idx, url in enumerate(links, 1):
        print(f"[INFO] 抓取详情 {idx}/{total}: {url}", flush=True)
        html = fetch_html(s, url, sleep=polite_sleep)
        papers.append(parse_paper_page(html, url))
    return papers


def generate_report_md(papers: List[Paper], summaries: Dict[str, Optional[str]],
                       keyword: str, since: datetime) -> str:
    def fmt_dt(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d")
    lines = []
    lines.append(f"# arXiv CS — 关键词“{keyword}”自 {fmt_dt(since)} 以来的最新论文\n")
    lines.append(f"共计 {len(papers)} 篇；排序：最新优先（按 announced_date_first）。\n")
    for i, p in enumerate(papers, 1):
        sumtext = summaries.get(p.url)
        if sumtext is None:
            sumtext = "_（未生成模型摘要；请设置 API Key 后重试）_"
        lines.append(f"## {i}. {p.title}\n")
        lines.append(f"- **链接**：{p.url}")
        lines.append(f"- **作者**：{', '.join(p.authors) if p.authors else '(unknown)'}")
        lines.append(f"- **提交时间**：{fmt_dt(p.submitted)}")
        lines.append(f"\n**摘要**：\n\n{p.abstract}\n")
        lines.append(f"**模型解读（{MODEL_NAME}）**：\n\n{sumtext}\n")
        lines.append("---\n")
    return "\n".join(lines)


def generate_report_json(papers: List[Paper], summaries: Dict[str, Optional[str]],
                         keyword: str, since: datetime) -> Dict:
    data = {
        "keyword": keyword,
        "since": since.strftime("%Y-%m-%d"),
        "model": MODEL_NAME,
        "base_url": os.environ.get("OPENAI_BASE_URL")
                   or os.environ.get("ARXIV_AGENT_BASE_URL")
                   or OPENAI_BASE_URL_DEFAULT,
        "papers": [],
    }
    for p in papers:
        data["papers"].append({
            "url": p.url,
            "arxiv_id": p.arxiv_id,
            "title": p.title,
            "authors": p.authors,
            "abstract": p.abstract,
            "submitted": p.submitted.strftime("%Y-%m-%d"),
            "model_summary": summaries.get(p.url),
        })
    return data


# ---------------- CLI ----------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="arXiv academic agent (async summarize, incremental, no-overwrite on 'no new')")
    ap.add_argument("--keywords", type=str, required=True, help="逗号分隔：例如 \"SPARSE AUTOENCODER, circuit analysis\"")
    ap.add_argument("--since", type=str, required=True, help="YYYY-MM-DD，例如 2025-08-01")
    ap.add_argument("--page-size", type=int, default=PAGE_SIZE_DEFAULT)
    ap.add_argument("--max-pages", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=1.0, help="抓取页面/逐论文请求的间隔秒（礼貌）")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--api-key", type=str, default=None, help="API Key；优先级：--api-key > ARXIV_AGENT_API_KEY > OPENAI_API_KEY")
    ap.add_argument("--outdir", type=str, default="./output")
    ap.add_argument("--sum-concurrency", type=int, default=4, help="模型并发数上限")
    ap.add_argument("--sum-sleep", type=float, default=0.0, help="每条模型调用后可选 sleep（秒），防止限流）")
    ap.add_argument("--state-dir", type=str, default="./state", help="增量状态存储目录（已收录 arXiv ID）")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # 关键词 & since
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    try:
        since_dt = dateparser.parse(args.since).replace(tzinfo=None)
    except Exception as e:
        print(f"[Error] since 日期无法解析：{args.since}: {e}", file=sys.stderr)
        return 2

    # API Key 检测（运行时解析）
    env_ak = os.environ.get("ARXIV_AGENT_API_KEY")
    env_ok = os.environ.get("OPENAI_API_KEY")
    resolved_api_key = args.api_key or env_ak or env_ok
    print(f"[INFO] API Key 检测：ARXIV_AGENT_API_KEY={'SET' if env_ak else 'EMPTY'}, "
          f"OPENAI_API_KEY={'SET' if env_ok else 'EMPTY'}, 使用={'SET' if resolved_api_key else 'EMPTY'}",
          flush=True)

    session = _requests_session()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)

    all_outputs = []
    date_stamp = _derive_date_stamp(args.outdir)

    for kw in keywords:
        safe_kw = _safe_kw(kw)
        seen_ids = _load_seen(args.state_dir, safe_kw)

        print(f"[INFO] 搜索关键词：{kw}（自 {since_dt.date()} 起）", flush=True)
        try:
            links = search_keyword(
                kw, since=since_dt, page_size=args.page_size,
                max_pages=args.max_pages, polite_sleep=args.sleep, session=session,
            )
            print(f"[INFO] 命中链接：{len(links)} 篇", flush=True)
            papers = collect_papers(links, polite_sleep=args.sleep, session=session)
            print(f"[INFO] 解析论文详情完成：{len(papers)} 篇", flush=True)
        except Exception as e:
            print(f"[ERROR] 关键词“{kw}”处理失败：{e}. 跳过该关键词，继续后续。", flush=True)
            continue

        # ---- 仅保留“未收录”的论文 ----
        new_papers: List[Paper] = [p for p in papers if p.arxiv_id and p.arxiv_id not in seen_ids]
        print(f"[INFO] 过滤后新增论文：{len(new_papers)} 篇（已收录：{len(papers) - len(new_papers)} 篇）", flush=True)

        md_path = os.path.join(args.outdir, f"{safe_kw}_{date_stamp}.md")
        json_path = os.path.join(args.outdir, f"{safe_kw}_{date_stamp}.json")

        # ---------- 无新增 ----------
        if len(new_papers) == 0:
            if os.path.exists(md_path) or os.path.exists(json_path):
                # 已有当日文件 -> 不覆盖
                print("[INFO] 今天已有输出文件，且没有新增论文；保持原文件不变。", flush=True)
                all_outputs.append({"keyword": kw, "markdown": md_path, "json": json_path, "no_new": True, "skipped_write": True})
                continue
            # 当日文件不存在 -> 写一份“没有最新的论文发表”
            no_new_text = f"# arXiv CS — 关键词“{kw}”自 {since_dt.strftime('%Y-%m-%d')} 以来的最新论文\n\n没有最新的论文发表\n"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(no_new_text)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "keyword": kw,
                    "since": since_dt.strftime("%Y-%m-%d"),
                    "model": MODEL_NAME,
                    "base_url": os.environ.get("OPENAI_BASE_URL")
                               or os.environ.get("ARXIV_AGENT_BASE_URL")
                               or OPENAI_BASE_URL_DEFAULT,
                    "papers": [],
                    "message": "没有最新的论文发表",
                    "no_new": True,
                }, f, ensure_ascii=False, indent=2)
            print("没有最新的论文发表（首次生成当日文件）", flush=True)
            all_outputs.append({"keyword": kw, "markdown": md_path, "json": json_path, "no_new": True})
            continue

        # ---------- 有新增 ----------
        # 1) 仅对“新增论文”做模型解读
        if args.no_summarize or not resolved_api_key:
            print("[INFO] 不进行模型解读（未提供 API Key 或指定 --no-summarize）。", flush=True)
            new_summaries: Dict[str, Optional[str]] = {p.url: None for p in new_papers}
        else:
            cfg = SummarizerConfig(concurrent=max(1, args.sum_concurrency))
            summarizer = AsyncOpenAISummarizer(resolved_api_key, cfg)
            new_summaries = asyncio.run(summarizer.summarize_many(new_papers, sleep_between=args.sum_sleep))

        # 2) 若当日文件已存在：读入旧 JSON，合并 去重 并排序（submitted 降序）
        existing_papers, existing_summaries = _load_existing_report(json_path)
        merged_papers: List[Paper] = []
        merged_summaries: Dict[str, Optional[str]] = {}

        if existing_papers:
            merged_papers.extend(existing_papers)
            merged_summaries.update(existing_summaries)
        merged_papers.extend(new_papers)
        merged_summaries.update(new_summaries)

        # 去重（按 arxiv_id 保序后再整体按 submitted 降序排序）
        dedup: List[Paper] = []
        seen_ids2: Set[str] = set()
        for p in merged_papers:
            aid = p.arxiv_id or ""
            if aid and aid not in seen_ids2:
                dedup.append(p)
                seen_ids2.add(aid)

        dedup.sort(key=lambda x: x.submitted, reverse=True)

        # 3) 生成当日文件（包含旧+新，覆盖写入）
        report_md = generate_report_md(dedup, merged_summaries, kw, since_dt)
        report_json = generate_report_json(dedup, merged_summaries, kw, since_dt)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_json, f, ensure_ascii=False, indent=2)

        # 4) 合并写回“已收录”集合
        for p in new_papers:
            seen_ids.add(p.arxiv_id)
        _save_seen(args.state_dir, safe_kw, seen_ids)

        print(f"[OK] 输出（合并写入）：\n  - {md_path}\n  - {json_path}", flush=True)
        all_outputs.append({"keyword": kw, "markdown": md_path, "json": json_path, "no_new": False})

    index_path = os.path.join(args.outdir, "INDEX.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print(f"[OK] 任务完成（索引：{index_path}）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
