from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERMES = ROOT / "vendor" / "hermes-agent"


class HermesVendorTests(unittest.TestCase):
    def test_fixed_release_metadata_and_license(self) -> None:
        metadata = tomllib.loads((HERMES / "pyproject.toml").read_text(encoding="utf-8"))
        project = metadata["project"]
        self.assertEqual(project["name"], "hermes-agent")
        self.assertEqual(project["version"], "0.17.0")
        self.assertEqual(project["license"], "MIT")

        license_text = (HERMES / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("MIT License", license_text)
        self.assertIn("Copyright (c) 2025 Nous Research", license_text)

    def test_required_release_files_exist(self) -> None:
        for relative in (
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "uv.lock",
            "setup-hermes.sh",
            "VENDORED-SOURCE.md",
        ):
            with self.subTest(relative=relative):
                self.assertTrue((HERMES / relative).is_file(), relative)

    def test_runtime_dependencies_and_state_are_not_vendored(self) -> None:
        forbidden_directories = {
            ".venv",
            "venv",
            "node_modules",
            ".playwright",
            ".pytest-cache",
            "__pycache__",
        }
        forbidden_files = {".env", ".install_method"}
        for path in HERMES.rglob("*"):
            relative = path.relative_to(HERMES)
            self.assertFalse(
                any(part in forbidden_directories for part in relative.parts),
                relative.as_posix(),
            )
            self.assertNotIn(path.name, forbidden_files, relative.as_posix())
            self.assertFalse(path.name.endswith(".pyc"), relative.as_posix())
            self.assertFalse(path.name.endswith(".egg-info"), relative.as_posix())

    def test_installer_has_version_gate_and_no_embedded_credentials(self) -> None:
        installer = (ROOT / "scripts" / "install-hermes.sh").read_text(encoding="utf-8")
        self.assertIn('EXPECTED_VERSION="0.17.0"', installer)
        self.assertIn("--dry-run", installer)
        self.assertIn("--apply", installer)
        self.assertIn("setup-hermes.sh", installer)
        self.assertNotRegex(installer, re.compile(r"sk-(?:proj-)?[A-Za-z0-9]{20,}"))


if __name__ == "__main__":
    unittest.main()
