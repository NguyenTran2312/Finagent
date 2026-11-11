import tempfile
import unittest
from pathlib import Path

from src.extraction.templates import (
    DEFAULT_TICKER_TEMPLATES,
    TemplateConfigError,
    TickerTemplates,
    load_ticker_templates,
)


class TickerTemplatesTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.tmp = self._dir.name

    def write(self, text, name="ticker_templates.yaml", data=None):
        path = Path(self.tmp) / name
        path.write_bytes(data if data is not None else text.encode("utf-8"))
        return path

    def test_load_valid(self):
        t = load_ticker_templates(self.write("ACB: bank\nvjc: corporate\nFTS: securities\n"))
        self.assertEqual(t.get("acb"), "bank")
        self.assertEqual(t.get("VJC"), "corporate")
        self.assertIsNone(t.get("XXX"))
        self.assertIn("fts", t)
        self.assertNotIn("XXX", t)
        self.assertEqual(t.calculation_templates(), {"ACB": "bank", "VJC": "corporate"})

    def test_errors(self):
        cases = {"- ACB\n": "-", "ACB: retail\n": "ACB", "ACB: bank\nacb: bank\n": "ACB",
                 "'  ': bank\n": "?", "1: bank\n": "?", "": "-", "ACB: 5\n": "ACB",
                 "acb: [bank]\n": "ACB", "ACB: [unclosed\n": "-"}
        for text, ticker in cases.items():
            with self.subTest(text=text), self.assertRaises(TemplateConfigError) as ctx:
                load_ticker_templates(self.write(text))
            self.assertEqual(ctx.exception.ticker, ticker)

    def test_invalid_utf8(self):
        with self.assertRaises(TemplateConfigError) as ctx:
            load_ticker_templates(self.write("", data=b"\xff\xfe\x00ACB: bank"))
        self.assertEqual(ctx.exception.ticker, "-")
        self.assertIn("UnicodeDecodeError", ctx.exception.reason)

    def test_message_format_and_no_raw_text(self):
        with self.assertRaises(TemplateConfigError) as ctx:
            load_ticker_templates(self.write("acb: secret-value\n"))
        e = ctx.exception
        self.assertEqual(str(e), f"{e.file}: {e.ticker}: {e.reason}")
        self.assertEqual(e.file, "ticker_templates.yaml")
        self.assertNotIn("secret-value", str(e))

    def test_missing_file(self):
        with self.assertRaises(TemplateConfigError) as ctx:
            load_ticker_templates(Path(self.tmp) / "absent.yaml")
        self.assertEqual(ctx.exception.ticker, "-")

    def test_constructor_and_non_string_lookup(self):
        t = TickerTemplates({"ACB": "bank"})
        self.assertEqual(t.get(" acb "), "bank")
        self.assertIsNone(t.get(None))
        self.assertNotIn(5, t)

    def test_default_path(self):
        self.assertEqual(DEFAULT_TICKER_TEMPLATES.name, "ticker_templates.yaml")
        self.assertEqual(DEFAULT_TICKER_TEMPLATES.parent.name, "config")


if __name__ == "__main__":
    unittest.main()
