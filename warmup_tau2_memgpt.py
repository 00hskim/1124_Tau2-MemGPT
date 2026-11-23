"""CHIMERA V5 ↔ tau2 + MemGPT core-memory (A-MEM 대체). TAU2_DOMAIN=retail|airline|telecom."""
import sys, os, json
E1 = "/home/hyunseo/[4]_Claude_Code_Stub/[Env_Setting_Clean]/[Env1]_tau2+A-MEM"
FW = "/home/hyunseo/[4]_Claude_Code_Stub/[TMP]/FrameWork_V5"
MEMDIR = "/home/hyunseo/[4]_Claude_Code_Stub/[TMP]/MemGPT_Mem"
DOMAIN = os.environ.get("TAU2_DOMAIN", "airline")
REP = os.environ.get("REP", "1")
LOGDIR = os.path.join("/home/hyunseo/[4]_Claude_Code_Stub/[18]_동계 학술대회/Warm_Up_Log2/tau2", DOMAIN, "rep"+REP)
STMDIR = os.path.join(LOGDIR, "stm"); os.makedirs(STMDIR, exist_ok=True)
os.environ.setdefault("TAU2_DATA_DIR", os.path.join(E1, "tau2-bench", "data"))
sys.path.insert(0, MEMDIR); sys.path.insert(0, os.path.join(E1, "integration")); sys.path.insert(0, FW)
KEY = os.environ["GOOGLE_API_KEY"]

from tau2_memgpt_agent import Tau2MemGPTAgent, write_conversation_to_core
from core_memory import CoreMemory
_envmod = __import__(f"tau2.domains.{DOMAIN}.environment", fromlist=["get_environment"])
get_environment = _envmod.get_environment
from tau2.data_model.message import UserMessage, MultiToolMessage
from chimera.generator import run_generator
from chimera.executor import run_attack
from chimera.config import REACT_MAX_STEPS
import chimera.logging_util as L
L.setup(os.path.join(LOGDIR, "llm_calls.jsonl")); L.reassert()


def make_agent(cm=None):
    env = get_environment()
    ag = Tau2MemGPTAgent(tools=env.get_tools(), domain_policy=env.get_policy(),
                         llm="gemini/gemini-2.5-flash",
                         llm_args={"temperature": 0.0, "reasoning_effort": "disable", "drop_params": True},
                         core_memory=cm)
    return ag, env


def target_fn(prompt, reset=True):
    ag, env = make_agent(cm=CoreMemory())
    st = ag.get_init_state([])
    r, _ = ag.generate_next_message(UserMessage(role="user", content=prompt), st)
    return r.content or ""


_cur = {}
def converse_factory(retrieve_query=None):
    is_poison = retrieve_query is None
    cm = _cur["cm"]
    ag, env = make_agent(cm=cm)
    ctx = {"st": ag.get_init_state([])}
    def converse(user_msg):
        msg = UserMessage(role="user", content=user_msg); tcs_all = []
        for _ in range(REACT_MAX_STEPS):
            reply, ctx["st"] = ag.generate_next_message(msg, ctx["st"])
            tcs = reply.tool_calls or []
            if not tcs:
                if is_poison:
                    try: write_conversation_to_core(cm, [UserMessage(role="user", content=user_msg)])
                    except Exception: pass
                return reply.content or "", tcs_all
            tcs_all += tcs
            results = [env.get_response(tc) for tc in tcs]
            msg = results[0] if len(results) == 1 else MultiToolMessage(role="tool", tool_messages=results)
        return reply.content or "", tcs_all
    return converse


print(f"=== [tau2/{DOMAIN}] MemGPT core-memory | profiling → 생성 ===")
out = run_generator(target_fn, only_strategies=["M1", "M3", "M6", "M7"])
json.dump(out, open(os.path.join(LOGDIR, "scenarios.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n=== 멀티턴 발사 (MemGPT) ===")
for sc in out["scenarios"]:
    if not sc.get("items") or not sc.get("trigger"):
        print(f"  {sc['strategy']}×{sc['harm']}: 스킵"); continue
    _cur["cm"] = CoreMemory()
    meta = {"strategy": sc["strategy"], "harm": sc["harm"], "domain": DOMAIN}
    try:
        run_attack(converse_factory, sc, {}, meta, session_dir=STMDIR)
        print(f"  {sc['strategy']}×{sc['harm']}: 발사완료")
    except Exception as e:
        print(f"  {sc['strategy']}×{sc['harm']}: 크래시 ({type(e).__name__})")
print("\nDONE_MARKER")
