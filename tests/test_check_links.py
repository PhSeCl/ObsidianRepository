import os
import shutil
import subprocess
import sys
import textwrap
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_links.py"
TEST_TMP_ROOT = REPO_ROOT / ".tmp-tests"


def run_check_links(root: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root)],
        capture_output=True,
        env=env,
        check=False,
    )


@contextmanager
def workspace_tmpdir():
    TEST_TMP_ROOT.mkdir(exist_ok=True)
    path = TEST_TMP_ROOT / f"tmp-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class CheckLinksTests(unittest.TestCase):
    def test_anchor_check_accepts_matching_heading_from_any_duplicate_basename(self) -> None:
        with workspace_tmpdir() as root:
            (root / "course-a").mkdir()
            (root / "course-b").mkdir()
            (root / "course-a" / "00_课程总览_MOC.md").write_text(
                "# 课程 A\n\n## Section X\n",
                encoding="utf-8",
            )
            (root / "course-b" / "00_课程总览_MOC.md").write_text(
                "# 课程 B\n\n## Other Section\n",
                encoding="utf-8",
            )
            (root / "quiz.md").write_text(
                "[[00_课程总览_MOC#Section X]]\n",
                encoding="utf-8",
            )

            result = run_check_links(root)

            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout.decode("utf-8", errors="replace"),
            )

    def test_reporting_does_not_crash_on_gbk_output_with_emoji_anchor(self) -> None:
        with workspace_tmpdir() as root:
            (root / "note.md").write_text(
                textwrap.dedent(
                    """\
                    [[missing#📊 成绩与考核]]
                    """
                ),
                encoding="utf-8",
            )

            result = run_check_links(
                root,
                extra_env={
                    "PYTHONIOENCODING": "gbk",
                    "PYTHONUTF8": "0",
                },
            )
            stdout = result.stdout.decode("gbk", errors="replace")
            stderr = result.stderr.decode("gbk", errors="replace")

            self.assertEqual(result.returncode, 1, msg=stdout + stderr)
            self.assertNotIn("UnicodeEncodeError", stdout + stderr)
            self.assertIn("问题合计: 1", stdout)


if __name__ == "__main__":
    unittest.main()
