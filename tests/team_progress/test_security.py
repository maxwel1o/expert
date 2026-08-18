import unittest

from team_progress.security import UnsafeTextError, validate_safe_text


class SecurityTests(unittest.TestCase):
    def test_allows_credential_alias(self):
        self.assertEqual(
            "use credential:model-server-admin",
            validate_safe_text("use credential:model-server-admin", "message"),
        )

    def test_rejects_secret_assignment(self):
        with self.assertRaises(UnsafeTextError):
            validate_safe_text("password=abcdefghijklmnop", "message")

    def test_rejects_private_key(self):
        with self.assertRaises(UnsafeTextError):
            validate_safe_text("-----BEGIN PRIVATE KEY-----", "message")

    def test_rejects_credential_url(self):
        with self.assertRaises(UnsafeTextError):
            validate_safe_text("https://admin:password123@example.invalid/x", "message")


if __name__ == "__main__":
    unittest.main()
