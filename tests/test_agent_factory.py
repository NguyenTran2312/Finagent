from pathlib import Path
import pkgutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import src.agent
from src.agent.agent import ReasoningAgent
from src.agent.factory import build_agent
from src.agent.llm import CountingLlm
from src.agent.types import AgentConfig
from src.extraction.extractor import LeafExtractor
from tests.agent_fixtures import ScriptedLlm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AgentFactoryTests(unittest.TestCase):
    def test_build_agent_without_retriever_touches_no_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            with mock.patch("src.retrieval.table_retriever.open_table_retriever") as opener:
                agent = build_agent(AgentConfig(retriever_mode="none"),
                                    llm=CountingLlm(ScriptedLlm()), db=missing / "corpus.sqlite3",
                                    bm25_dir=missing / "bm25", chroma_dir=missing / "chroma")
                self.assertIsInstance(agent, ReasoningAgent)
                self.assertIsInstance(agent._extractor_factory(None), LeafExtractor)
                opener.assert_not_called()

    def test_retriever_opens_once_and_lazily_for_bm25(self):
        sentinel = object()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("src.retrieval.table_retriever.open_table_retriever",
                            return_value=sentinel) as opener:
                agent = build_agent(AgentConfig(retriever_mode="bm25"),
                                    llm=CountingLlm(ScriptedLlm()), db=Path(tmp) / "c.sqlite3",
                                    bm25_dir=Path(tmp) / "bm25", chroma_dir=None)
                opener.assert_not_called()
                first, second = agent._extractor_factory(None), agent._extractor_factory(None)
                self.assertEqual(opener.call_count, 1)
                self.assertIs(first._retriever, sentinel)
                self.assertIs(second._retriever, sentinel)
                self.assertEqual(opener.call_args.kwargs["mode"], "bm25")

    def test_agent_package_imports_are_light(self):
        names = ["src.agent"] + [m.name for m in pkgutil.iter_modules(src.agent.__path__, "src.agent.")]
        self.assertIn("src.agent.factory", names)
        code = (
            "import importlib, sys\n"
            f"for name in {names!r}:\n    importlib.import_module(name)\n"
            "heavy = [m for m in ('streamlit', 'torch', 'chromadb', 'langchain_openai') if m in sys.modules]\n"
            "assert not heavy, heavy\n"
        )
        env_path = f"{PROJECT_ROOT}{';' if sys.platform == 'win32' else ':'}{PROJECT_ROOT / 'src'}"
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, env={**__import__("os").environ, "PYTHONPATH": env_path})
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
