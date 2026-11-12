"""Tests for retrieval-guided leaf extraction."""

from __future__ import annotations

import dataclasses
import unittest

from src.extraction.outcomes import ExtractedFact
from src.extraction.retrieval import RankedTable, ranking_of


class RankingValidationTests(unittest.TestCase):
    def test_valid_answer_maps_chunks_to_ranks(self):
        answer = (RankedTable("D:table:2", 2), RankedTable("D:table:1", 1))
        self.assertEqual(ranking_of(answer), {"D:table:2": 2, "D:table:1": 1})
        self.assertEqual(ranking_of(()), {})

    def test_invalid_answers_raise(self):
        bad = {
            "list": [RankedTable("a", 1)],
            "rank_zero": (RankedTable("a", 0),),
            "rank_gap": (RankedTable("a", 1), RankedTable("b", 3)),
            "duplicate_rank": (RankedTable("a", 1), RankedTable("b", 1)),
            "duplicate_chunk": (RankedTable("a", 1), RankedTable("a", 2)),
            "not_ranked_table": (("a", 1),),
        }
        for name, answer in bad.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    ranking_of(answer)

    def test_extracted_fact_has_no_trace_by_default(self):
        self.assertIsNone(dataclasses.fields(ExtractedFact)[-1].default)
        self.assertEqual(dataclasses.fields(ExtractedFact)[-1].name, "retrieval")


if __name__ == "__main__":
    unittest.main()
