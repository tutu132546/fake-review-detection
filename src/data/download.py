"""带断点续传与多镜像回退的下载器。

这个模块存在的理由：本项目所在网络到 GitHub 直连不稳定（常见 502/连接被重置、
大文件传输中途截断），而数据集又动辄几百 MB。因此下载必须满足三条：
1. **可续传**：中断后用 Range 续上，而不是从头再来；
2. **可校验**：落盘前比对大小与 sha256，截断的文件必须被识别出来；
3. **可回退**：同一条数据配多个镜像地址，前一个失败自动换下一个。
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_CHUNK = 1024 * 1024  # 1 MiB
_USER_AGENT = "fake-review-detection/1.0 (course project; data acquisition)"


@dataclass(frozen=True)
class RemoteFile:
    """一个待下载文件：目标相对路径 + 一组候选地址 + 完整性信息。"""

    relpath: str
    urls: List[str]
    size: Optional[int] = None
    sha256: Optional[str] = None
    note: str = ""
    #: GitHub API 的 raw 下载需要专门的 Accept 头
    headers: dict = field(default_factory=dict)


def sha256_of(path: Path, chunk: int = _CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _existing_ok(dest: Path, spec: RemoteFile) -> bool:
    if not dest.exists():
        return False
    if spec.size is not None and dest.stat().st_size != spec.size:
        return False
    return True


def download_one(
    spec: RemoteFile,
    dest_dir: Path,
    *,
    retries: int = 3,
    timeout: tuple = (20, 120),
    force: bool = False,
) -> Path:
    """下载单个文件到 `dest_dir/relpath`，返回最终路径。

    已存在且大小/sha256 匹配则直接跳过；存在 `.part` 则续传。
    """
    dest = dest_dir / spec.relpath
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not force and _existing_ok(dest, spec):
        if spec.sha256 and sha256_of(dest) != spec.sha256:
            logger.warning("%s 大小正确但 sha256 不匹配，重新下载", dest.name)
        else:
            logger.info("跳过（已存在且完整）: %s", spec.relpath)
            return dest

    part = dest.with_suffix(dest.suffix + ".part")
    errors: List[str] = []

    for url in spec.urls:
        headers = {"User-Agent": _USER_AGENT, **spec.headers}
        # GitHub 的内容 API 需要显式要求原始字节，否则返回 base64 的 JSON 包装
        if "api.github.com" in url and "Accept" not in headers:
            headers["Accept"] = "application/vnd.github.raw"
        for attempt in range(1, retries + 1):
            try:
                downloaded = _fetch(url, part, headers=headers, timeout=timeout)
                if spec.size is not None and downloaded != spec.size:
                    raise IOError(f"大小不符：得到 {downloaded} 字节，期望 {spec.size}")
                if spec.sha256:
                    got = sha256_of(part)
                    if got != spec.sha256:
                        raise IOError(f"sha256 不符：得到 {got}")
                os.replace(part, dest)
                logger.info("下载完成: %s (%d 字节) 来源 %s", spec.relpath, downloaded, urlparse(url).netloc)
                return dest
            except Exception as exc:  # noqa: BLE001 - 需要捕获所有网络异常并换镜像
                errors.append(f"{url} 第 {attempt} 次失败: {exc}")
                logger.warning("%s 第 %d/%d 次失败: %s", urlparse(url).netloc, attempt, retries, exc)
                time.sleep(min(2 ** attempt, 10))

    detail = "\n  ".join(errors[-6:])
    if part.exists():
        logger.warning("保留未完成的临时文件以便续传: %s", part)
    raise RuntimeError(f"文件 {spec.relpath} 所有镜像均下载失败：\n  {detail}")


def _fetch(url: str, part: Path, *, headers: dict, timeout: tuple) -> int:
    """把 url 流式写入 part，支持 Range 续传，返回本次结束后的文件总字节数。"""
    resume_from = part.stat().st_size if part.exists() else 0
    req_headers = dict(headers)
    if resume_from:
        req_headers["Range"] = f"bytes={resume_from}-"
        logger.info("尝试续传 %s，从 %d 字节开始", part.name, resume_from)

    with requests.get(url, headers=req_headers, stream=True, timeout=timeout, allow_redirects=True) as resp:
        if resp.status_code == 416:  # 已经完整
            return part.stat().st_size
        if resp.status_code not in (200, 206):
            raise IOError(f"HTTP {resp.status_code}")
        # 服务器忽略了 Range，返回 200，只能从头写
        mode = "ab" if (resume_from and resp.status_code == 206) else "wb"
        if mode == "wb":
            resume_from = 0
        total = 0
        with open(part, mode) as f:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)
        return resume_from + total


def download_many(
    specs: Iterable[RemoteFile],
    dest_dir: Path,
    *,
    force: bool = False,
    tolerate_failure: bool = False,
) -> dict:
    """批量下载，返回 {relpath: "ok" | 错误信息}。

    `tolerate_failure=True` 时单个文件失败不中断（用于"能拿几个算几个"的场景），
    失败项会原样记进返回值和日志，绝不静默跳过。
    """
    report: dict = {}
    for spec in specs:
        try:
            download_one(spec, dest_dir, force=force)
            report[spec.relpath] = "ok"
        except Exception as exc:  # noqa: BLE001
            report[spec.relpath] = str(exc)
            logger.error("下载失败: %s -> %s", spec.relpath, exc)
            if not tolerate_failure:
                raise
    return report
