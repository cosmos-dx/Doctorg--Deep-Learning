"""
Multi-agent system for medical consultation.
"""

from app.agents.base import BaseAgent
from app.agents.orchestrator import AgentOrchestrator

__all__ = [
    "BaseAgent",
    "AgentOrchestrator",
]
