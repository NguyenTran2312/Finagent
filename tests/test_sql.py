import os
import unittest
from unittest.mock import patch

from config.settings import DATABASE_CONFIG, postgres_connection_params


class SqlConfigurationTests(unittest.TestCase):
    def test_postgres_connection_uses_configured_database(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            params = postgres_connection_params()

        self.assertEqual(params["host"], DATABASE_CONFIG["host"])
        self.assertEqual(params["port"], DATABASE_CONFIG["port"])
        self.assertEqual(params["dbname"], DATABASE_CONFIG["name"])
        self.assertEqual(params["user"], DATABASE_CONFIG["user"])
        self.assertNotIn("password", params)

    def test_postgres_connection_reads_overrides_and_password_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PGHOST": "127.0.0.1",
                "PGPORT": "5433",
                "PGDATABASE": "test_database",
                "PGUSER": "test_user",
                "POSTGRES_PASSWORD": "test-password",
            },
            clear=True,
        ):
            params = postgres_connection_params()

        self.assertEqual(params["host"], "127.0.0.1")
        self.assertEqual(params["port"], 5433)
        self.assertEqual(params["dbname"], "test_database")
        self.assertEqual(params["user"], "test_user")
        self.assertEqual(params["password"], "test-password")


if __name__ == "__main__":
    unittest.main()
