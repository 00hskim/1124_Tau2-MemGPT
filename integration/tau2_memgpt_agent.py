"""tau2 + MemGPT core-memory 배선 (A-MEM 대체).

tau2 내부 미수정. LLMAgent 서브클래싱해서, A-MEM 검색주입 대신
**MemGPT core-memory 블록을 시스템프롬프트에 상주 렌더**(검색 단계 없음).
저장 = 대화를 core-memory에 append(자율관리 메모리가 사실을 축적하는 것과 동형).
A-MEM과의 유일 차이 = top-k 검색 병목 제거(값이 무조건 컨텍스트에 실림).
"""
from __future__ import annotations
from typing import Any, List, Optional
import os, sys

from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import SystemMessage

sys.path.insert(0, os.path.dirname(__file__))
from core_memory import CoreMemory


class Tau2MemGPTAgent(LLMAgent):
    """LLMAgent + MemGPT core-memory 상주 렌더. 그 외 tau2 원본 동작 그대로."""

    def __init__(self, *args, core_memory: Optional[CoreMemory] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.core_memory = core_memory if core_memory is not None else CoreMemory()

    def get_init_state(self, message_history: Optional[list] = None) -> LLMAgentState:
        state = super().get_init_state(message_history)
        block = self.core_memory.render()          # 검색 없음 — 항상 상주 블록
        if block and state.system_messages:
            base = state.system_messages[0].content
            state.system_messages[0] = SystemMessage(role="system", content=base + "\n\n" + block)
        return state


def write_conversation_to_core(cm: CoreMemory, messages: list) -> int:
    """대화의 user 발화를 core-memory 'human' 블록에 append(사실 축적). assistant는 제외(노이즈 억제)."""
    if cm is None:
        return 0
    n = 0
    for m in messages:
        role = getattr(m, "role", "") or ""
        content = getattr(m, "content", None)
        if content and role == "user":
            cm.append("human", content)
            n += 1
    return n
