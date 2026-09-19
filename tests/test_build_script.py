"""Check platform-specific build finalization without compiling an extension."""

from pathlib import Path
import re
import subprocess
import unittest


class BuildFinalizationTest(unittest.TestCase):
    def run_finalizer(self, setup):
        source = (Path(__file__).resolve().parents[1] / "build.sh").read_text()
        helper = re.search(r"^fix_macos_libomp\(\) \{.*?^\}", source, re.M | re.S).group()
        result = subprocess.run(
            ["/bin/bash", "-c", "set -eu\n" + helper + "\n" + setup
             + "\nfix_macos_libomp\nprintf 'build completed'\n"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "build completed")

    def test_linux_build_reaches_success_after_optional_macos_fix(self):
        self.run_finalizer("PLATFORM=Linux")

    def test_missing_macos_tool_is_not_a_build_failure(self):
        self.run_finalizer("PLATFORM=Darwin\nPATH=/nonexistent")

    def test_missing_torch_openmp_is_not_a_build_failure(self):
        self.run_finalizer("""
PLATFORM=Darwin
install_name_tool() { return 99; }
python_without_openmp() { return 0; }
PYTHON_BIN=python_without_openmp
""")


if __name__ == "__main__":
    unittest.main()
