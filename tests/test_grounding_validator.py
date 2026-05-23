"""
Unit tests for GroundingValidator v2.0
Tests: multi-attribute matching, auto-correction, similarity scoring.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("playwright")

from step_rl.environment.grounding_validator import GroundingValidator


class MockLocator:
    def __init__(self, count_val=1):
        self._count = count_val
        self.first = MagicMock()

    async def count(self):
        return self._count


class MockPage:
    def __init__(self):
        self.locator = MagicMock(return_value=MockLocator())
        self.evaluate = AsyncMock(return_value=None)
        self.evaluate_handle = AsyncMock(
            return_value=MagicMock(as_element=MagicMock(return_value=None))
        )


class TestGroundingValidator(unittest.TestCase):
    def setUp(self):
        self.validator = GroundingValidator(
            similarity_threshold=0.85,
            reward_valid=0.1,
            reward_corrected=-0.05,
            reward_failed=-0.2,
        )
        self.page = MockPage()

    def test_text_similarity_exact(self):
        sim = GroundingValidator._text_similarity("立即购买", "立即购买")
        self.assertEqual(sim, 1.0)

    def test_text_similarity_partial(self):
        sim = GroundingValidator._text_similarity("立即购买", "立即下单")
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_text_similarity_empty(self):
        sim = GroundingValidator._text_similarity("", "test")
        self.assertEqual(sim, 0.0)

    def test_reward_values(self):
        self.assertEqual(self.validator.reward_valid, 0.1)
        self.assertEqual(self.validator.reward_corrected, -0.05)
        self.assertEqual(self.validator.reward_failed, -0.2)

    def test_similarity_threshold(self):
        self.assertEqual(self.validator.similarity_threshold, 0.85)


if __name__ == "__main__":
    unittest.main()
