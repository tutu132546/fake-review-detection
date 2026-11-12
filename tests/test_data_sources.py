"""下载清单与校验码的单元测试。

这里锁死两件事：
1. **每个远端文件都必须有写死的 sha256**——没有校验码的下载等于"代理给什么就信什么"；
2. **本地文件与官方仓库逐字节一致**（git blob sha1 复核），从而证明 sha256 可信。

第 2 条只在本地已下载对应文件时才跑（CI/队友机器没下数据也能通过）。

    python -m unittest discover -s tests -p "test_data_sources.py" -v
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import sources  # noqa: E402
from data.paths import RAW_DIR  # noqa: E402


def _git_blob_sha1(path: Path) -> str:
    """按 git 对象格式计算 sha1：sha1("blob <size>\\0" + 内容)。"""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class TestDownloadManifest(unittest.TestCase):
    def test_every_remote_file_has_checksum(self):
        missing = [
            spec.relpath
            for specs in sources.FILES.values()
            for spec in specs
            if not spec.sha256
        ]
        self.assertEqual(missing, [], f"以下文件没有写死 sha256：{missing}")

    def test_checksums_are_well_formed(self):
        for relpath, digest in sources.all_pinned_checksums().items():
            self.assertEqual(len(digest), 64, f"{relpath} 的 sha256 长度不对")
            int(digest, 16)  # 非十六进制会抛 ValueError

    def test_same_relpath_has_same_checksum(self):
        """同一个文件在多个数据集下重复登记时，校验码必须一致。"""
        seen: dict = {}
        for specs in sources.FILES.values():
            for spec in specs:
                if spec.relpath in seen and seen[spec.relpath] != spec.sha256:
                    self.fail(f"{spec.relpath} 在不同数据集下登记了不同的 sha256")
                seen[spec.relpath] = spec.sha256

    def test_urls_are_https(self):
        for specs in sources.FILES.values():
            for spec in specs:
                for url in spec.urls:
                    self.assertTrue(url.startswith("https://"), f"{spec.relpath} 的地址不是 https: {url}")

    def test_official_blobs_cover_all_fraudsquad_files(self):
        """官方 git blob 记录必须覆盖 fraudsquad 目录下的每个文件，否则校验有盲区。"""
        names = {rp.rsplit("/", 1)[-1] for rp in sources.all_pinned_checksums() if rp.startswith("fraudsquad/")}
        missing = names - set(sources.OFFICIAL_GIT_BLOBS)
        self.assertEqual(missing, set(), f"这些文件没有官方 blob 记录：{sorted(missing)}")


class TestLocalFilesMatchOfficial(unittest.TestCase):
    """本地已下载的文件必须与官方仓库逐字节一致（无数据时自动跳过）。"""

    def test_sha256_matches_pinned(self):
        checked = 0
        for specs in sources.FILES.values():
            for spec in specs:
                path = RAW_DIR / spec.relpath
                if not path.exists() or not spec.sha256:
                    continue
                with self.subTest(file=spec.relpath):
                    self.assertEqual(_sha256(path), spec.sha256, f"{spec.relpath} 与写死的 sha256 不符")
                    checked += 1
        if checked == 0:
            self.skipTest("本地没有已下载的受校验文件")

    def test_git_blob_matches_official(self):
        checked = 0
        for name, official in sources.OFFICIAL_GIT_BLOBS.items():
            path = RAW_DIR / "fraudsquad" / name
            if not path.exists():
                continue
            with self.subTest(file=name):
                self.assertEqual(_git_blob_sha1(path), official, f"{name} 与官方仓库字节不一致")
                checked += 1
        if checked == 0:
            self.skipTest("本地没有 FraudSquad 原始文件")


if __name__ == "__main__":
    unittest.main()
