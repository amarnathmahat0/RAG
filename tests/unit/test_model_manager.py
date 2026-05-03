"""Unit tests for ModelManager RAM guardrails (Ollama mocked)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.utils.exceptions import RAMGuardrailError
from src.utils.model_manager import ModelManager


class TestRAMGuardrail:
    def test_check_ram_raises_when_insufficient(self):
        mm = ModelManager()
        # Simulate only 500 MB free
        with patch.object(mm, "free_ram_mb", return_value=500.0):
            with pytest.raises(RAMGuardrailError):
                mm._check_ram(required_mb=100.0)  # 500 - 100 = 400 < 1500

    def test_check_ram_passes_when_sufficient(self):
        mm = ModelManager()
        with patch.object(mm, "free_ram_mb", return_value=4000.0):
            # Should not raise: 4000 - 100 = 3900 > 1500
            mm._check_ram(required_mb=100.0)

    def test_check_ram_edge_case(self):
        mm = ModelManager()
        # Exactly at boundary: free=1600, required=100, buffer=1500 → 1600-100=1500 == 1500 → pass
        with patch.object(mm, "free_ram_mb", return_value=1600.0):
            mm._check_ram(required_mb=100.0)

    def test_free_ram_returns_positive(self):
        mm = ModelManager()
        assert mm.free_ram_mb() > 0

    def test_total_ram_returns_positive(self):
        mm = ModelManager()
        assert mm.total_ram_mb() > 0


@pytest.mark.asyncio
class TestModelManagerStatus:
    async def test_status_structure(self):
        mm = ModelManager()
        status = await mm.status()
        assert "loaded_models" in status
        assert "free_ram_mb" in status
        assert "total_ram_mb" in status
        assert "guardrail_mb" in status
        assert isinstance(status["loaded_models"], list)
