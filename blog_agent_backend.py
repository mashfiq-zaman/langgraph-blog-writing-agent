from __future__ import annotations

import operator
from typing import TypedDict, List, Annotated, Literal, Optional

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()


class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3–5 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(
        ...,
        description="Target word count for this section (120–450).",
    )
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

   
class Plan(BaseModel):
    blog_title:str
    tasks: List[Task]
    audience: str= Field(..., description='who this blog is for')
    tone: str = Field(..., description=' writing tone eg., practical, crisp')
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)

class evidenceitem(BaseModel):
    title: str
    url : str
    published_at: Optional[str]=None
    snippet: Optional[str]=None
    source: Optional[str]=None

class routerdecision(BaseModel):
    need_research: bool
    mode: Literal['closed_book', 'hybrid', 'open_book']
    queries: List[str]= Field(default_factory= list)

class evidencepack(BaseModel):
    evidence: list[evidenceitem]=Field(default_factory= list)

class imagespec(BaseModel):
    placeholder: str = Field(..., description='e.g. [[IMAGE_1]]')
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[imagespec] = Field(default_factory=list)




class state(TypedDict):
    topic: str

    mode:str
    need_research: bool
    queries: List[str]
    evidence: List[evidenceitem]


    plan: Optional[Plan]
    as_of: str
    recency_days:int

    merged_md:str
    md_with_placeholders: str
    image_specs:List[dict]
    sections: Annotated[List[str], operator.add]
    final:str


llm=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')


# Router (decide upfront)

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def router_node(state: state):

    topic= state['topic']
    decider= llm.with_structured_output(routerdecision)
    decision= decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f'Topic : {topic}')
        ]
    )

    return {
        'need_research': decision.need_research,
        'mode':decision.mode,
        'queries':decision.queries,

    }
def route_next(state: state):
    return 'research' if state['need_research'] else 'orchestrator'


### research
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""
def research_node(state: state):
    ##takes the first 10 queries of the state
    queries= (state.get('queries',[])or [])
    max_result=6

    raw_results: List[dict]=[]

    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_result))

    if not  raw_results:
        return {'evidence':[]}

    extractor= llm.with_structured_output(evidencepack)

    pack= extractor.invoke(
       [ SystemMessage(content=RESEARCH_SYSTEM),
    HumanMessage(content=f'Raw results:\n{raw_results}')
    ])

    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}


def orchestrator(state: state) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    plan= llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(content=(
                    "You are a senior technical writer and developer advocate. Your job is to produce a "
                    "highly actionable outline for a technical blog post.\n\n"
                    "Hard requirements:\n"
                    "- Create 5–7 sections (tasks) that fit a technical blog.\n"
                    "- Each section must include:\n"
                    "  1) goal (1 sentence: what the reader can do/understand after the section)\n"
                    "  2) 3–5 bullets that are concrete, specific, and non-overlapping\n"
                    "  3) target word count (120–450)\n"
                    "- Include EXACTLY ONE section with section_type='common_mistakes'.\n\n"
                    "Make it technical (not generic):\n"
                    "- Assume the reader is a developer; use correct terminology.\n"
                    "- Prefer design/engineering structure: problem → intuition → approach → implementation → "
                    "trade-offs → testing/observability → conclusion.\n"
                    "- Bullets must be actionable and testable (e.g., 'Show a minimal code snippet for X', "
                    "'Explain why Y fails under Z condition', 'Add a checklist for production readiness').\n"
                    "- Explicitly include at least ONE of the following somewhere in the plan (as bullets):\n"
                    "  * a minimal working example (MWE) or code sketch\n"
                    "  * edge cases / failure modes\n"
                    "  * performance/cost considerations\n"
                    "  * security/privacy considerations (if relevant)\n"
                    "  * debugging tips / observability (logs, metrics, traces)\n"
                    "- Avoid vague bullets like 'Explain X' or 'Discuss Y'. Every bullet should state what "
                    "to build/compare/measure/verify.\n\n"
                    "Ordering guidance:\n"
                    "- Start with a crisp intro and problem framing.\n"
                    "- Build core concepts before advanced details.\n"
                    "- Include one section for common mistakes and how to avoid them.\n"
                    "- End with a practical summary/checklist and next steps.\n\n"
                    "Output must strictly match the Plan schema."
                )
            ),

            HumanMessage(content=
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}")
        ]
    )
    return {'plan':plan}


#checks how many task in plan object and thne trigger each worker node
def fanout(state: state):
    return [Send('worker',{'task': task.model_dump(),'topic': state['topic'],'plan': state['plan'].model_dump(),'mode':state['mode'],'evidence':[e.model_dump() for e in state.get('evidence', [])]})
                 
            for task in state['plan'].tasks]


#from fanout to worker for each task it will trigger each worker
MERMAID_SYSTEM = """Return ONLY valid Mermaid diagram code (flowchart, sequenceDiagram, etc.).
Short labels, max ~12 nodes. No code fences, no explanation."""

def _mermaid_block(spec: dict) -> str:
    resp = llm.invoke([SystemMessage(content=MERMAID_SYSTEM),
                       HumanMessage(content=spec["prompt"])])
    code = to_text(resp.content).strip().strip("`")
    code = re.sub(r"^mermaid\s*\n", "", code)   # drop a stray 'mermaid' tag
    return f"```mermaid\n{code}\n```\n*{spec['caption']}*"
def to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""
def worker(payload: dict) ->dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [evidenceitem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )
    response = llm.invoke([
        SystemMessage(content=(
        """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

    )
)
,
        HumanMessage(content=(
            f"Blog: {plan.blog_title}\n"
                     f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
        ))
    ])

    section_md = to_text(response.content).strip()

    return {'sections':  [(task.id, section_md)]}


import re
from pathlib import Path

def merge_content(state: state) -> dict:
    plan = state["plan"]
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    return {"merged_md": f"# {plan.blog_title}\n\n{body}\n"}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if diagrams are needed for THIS blog.

Rules:
- Max 3 diagrams total.
- Each diagram must materially improve understanding (flow, architecture, comparison).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no diagrams needed: md_with_placeholders must equal input and images=[].
- The 'prompt' field must describe a diagram: list the nodes and the arrows between them.
Return strictly GlobalImagePlan.
"""

def decide_images(state: state) -> dict:
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]

    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=(
            f"Blog kind: {plan.blog_kind}\n"
            f"Topic: {state['topic']}\n\n"
            "Insert placeholders + propose diagram descriptions.\n\n"
            f"{merged_md}"
        )),
    ])

    md = image_plan.md_with_placeholders
    if len(md) < 0.8 * len(merged_md):        # model shortened the blog -> ignore it
        return {"md_with_placeholders": merged_md, "image_specs": []}

    return {
        "md_with_placeholders": md,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


MERMAID_SYSTEM = """Return ONLY valid Mermaid diagram code (flowchart, sequenceDiagram, etc.).
Short labels, max ~12 nodes. No code fences, no explanation."""

def _mermaid_block(spec: dict) -> str:
    resp = llm.invoke([SystemMessage(content=MERMAID_SYSTEM),
                       HumanMessage(content=spec["prompt"])])
    code = to_text(resp.content).strip().strip("`")
    code = re.sub(r"^mermaid\s*\n", "", code)
    return f"```mermaid\n{code}\n```\n*{spec['caption']}*"


def safe_filename(title: str, ext: str = ".md") -> str:
    name = re.sub(r'[<>:"/\\|?*\r\n\t]', "", title).strip().rstrip(".")
    name = re.sub(r"\s+", "_", name)[:100] or "blog"
    return name + ext


def generate_and_place_images(state: state) -> dict:
    plan = state["plan"]
    md = state.get("md_with_placeholders") or state["merged_md"]

    for spec in state.get("image_specs") or []:
        try:
            md = md.replace(spec["placeholder"], _mermaid_block(spec))
        except Exception as e:
            print(f"Diagram failed for {spec['placeholder']}: {e}")
            md = md.replace(spec["placeholder"], "")

    filename = safe_filename(plan.blog_title)
    Path(filename).write_text(md, encoding="utf-8")
    print("Saved:", filename)
    return {"final": md}



# build reducer subgraph
reducer_graph = StateGraph(state)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

reducer_subgraph


graph= StateGraph(state)

graph.add_node('router', router_node)
graph.add_node('research', research_node)
graph.add_node('orchestrator', orchestrator)
graph.add_node('worker',worker)
graph.add_node('reducer',reducer_subgraph)

graph.add_edge(START,'router')
graph.add_conditional_edges('router', route_next,{'research': 'research','orchestrator':'orchestrator'})

graph.add_edge('research','orchestrator')

graph.add_conditional_edges('orchestrator', fanout, ['worker'])
graph.add_edge('worker','reducer')
graph.add_edge('reducer',END)

app=graph.compile()


app