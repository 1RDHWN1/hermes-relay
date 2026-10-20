"""Provider adapters for the relay realtime-agent broker."""

from __future__ import annotations

from .base import RealtimeAgentProviderAdapter
from .gemini import GeminiLiveRealtimeAgentAdapter, GeminiLiveRealtimeAgentProvider
from .openai import OpenAIRealtimeAgentAdapter, OpenAIRealtimeAgentProvider
from .xai import XAIRealtimeAgentAdapter, XAIRealtimeAgentProvider


def adapter_for(provider_id: str) -> RealtimeAgentProviderAdapter:
    if provider_id == OpenAIRealtimeAgentAdapter.provider_id:
        return OpenAIRealtimeAgentAdapter()
    if provider_id == XAIRealtimeAgentAdapter.provider_id:
        return XAIRealtimeAgentAdapter()
    if provider_id == GeminiLiveRealtimeAgentAdapter.provider_id:
        return GeminiLiveRealtimeAgentAdapter()
    return RealtimeAgentProviderAdapter(provider_id)


__all__ = [
    "GeminiLiveRealtimeAgentAdapter",
    "GeminiLiveRealtimeAgentProvider",
    "OpenAIRealtimeAgentAdapter",
    "OpenAIRealtimeAgentProvider",
    "RealtimeAgentProviderAdapter",
    "XAIRealtimeAgentAdapter",
    "XAIRealtimeAgentProvider",
    "adapter_for",
]
