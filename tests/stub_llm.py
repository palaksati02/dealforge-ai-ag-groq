"""A scripted LLM for tests: lets the REAL CrewAI machinery (Crew.kickoff, Task guardrails,
Pydantic output parsing, ReAct tool loop) run deterministically with no network or API key."""
from typing import Any, Callable, List, Optional

from crewai.llms.base_llm import BaseLLM


def _text_of(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(str(m.get("content", "")) for m in messages)


class ScriptedLLM(BaseLLM):
    responder: Any = None          # callable(full_prompt_text, call_index) -> str
    calls: int = 0
    prompts: List[str] = []

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        text = _text_of(messages)
        self.prompts.append(text)
        idx = self.calls
        self.calls += 1
        out = self.responder(text, idx)
        # Report deterministic "provider" usage so token/cost accounting can be tested end-to-end.
        self._track_token_usage_internal({
            "prompt_tokens": max(1, len(text) // 4), "completion_tokens": max(1, len(str(out)) // 4),
            "total_tokens": max(1, len(text) // 4) + max(1, len(str(out)) // 4), "successful_requests": 1,
        })
        return out

    def supports_function_calling(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8000


def scripted(responder: Callable[[str, int], str]) -> ScriptedLLM:
    return ScriptedLLM(model="scripted-test-llm", responder=responder, prompts=[])


def workflow_router_llm() -> ScriptedLLM:
    """Scripted LLM that answers all three LLM-backed CrewAI agents with schema-valid output.
    Used by tests and by the evaluation script's token/cost ESTIMATE (real CrewAI-built prompts)."""
    import json
    import re
    state = {"coord": 0}

    def respond(prompt: str, _i: int) -> str:
        if "Extract the purchase requirement" in prompt:
            text = re.search(r"request text:\n---\n(.*?)\n---", prompt, re.S).group(1)
            from app.agents.procurement_crew import ProcurementWorkflowEngine
            ref = ProcurementWorkflowEngine(None).run_requirement_agent(text)
            return json.dumps({**ref.model_dump(mode="json"), "budget_limit": float(ref.budget_limit)})
        if "counter-offer to vendor" in prompt:
            price = re.search(r"target price is (₹[\d,]+\.\d{2})", prompt).group(1)
            return json.dumps({
                "justification": "Enterprise volume commitment and prompt payment terms justify a more competitive rate.",
                "draft_message": f"Dear Sales Team, thank you for your quotation. Given our volume and prompt payment our target price is {price}. Kind regards, DealForge Procurement"})
        if "request_adk_risk_assessment" in prompt:
            state["coord"] += 1
            if state["coord"] % 2 == 1:
                payload_json = re.search(r"exact JSON string \(do not alter it\):\n(\{.*?\})\n", prompt, re.S).group(1)
                return ("Thought: I need the independent assessment.\nAction: request_adk_risk_assessment\n"
                        "Action Input: " + json.dumps({"payload_json": payload_json}))
            return 'Thought: I now know the final answer\nFinal Answer: {"summary": "The ADK agent assessed the deal and returned its recommendation."}'
        raise AssertionError("unexpected prompt")

    return scripted(respond)
