# LangGraph Migration Plan

## Status: Future (migrate after LangChain pipeline is stable)

## Why Migrate?
- LangGraph supports stateful, cyclic agent workflows
- Better support for parallel agent execution with state management
- Easier to add conditional re-runs (e.g., re-run biological agent if score is borderline)

## Migration Strategy
1. Keep all agent `run()` functions unchanged — they are framework-agnostic
2. Replace LangChain `AgentExecutor` with a LangGraph `StateGraph`
3. Define a `PipelineState` TypedDict with all agent outputs
4. Each agent becomes a LangGraph node

## Basic LangGraph Structure (to be fleshed out)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

class PipelineState(TypedDict):
    image_path: str
    preprocessing: Optional[dict]
    geometry: Optional[dict]
    frequency: Optional[dict]
    texture: Optional[dict]
    biological: Optional[dict]
    vlm: Optional[dict]
    fusion: Optional[dict]
    report_path: Optional[str]

# Each agent becomes a node:
# graph.add_node("preprocessing", preprocessing_node)
# graph.add_node("geometry", geometry_node)
# etc.

# Edges define execution order and parallelism
```

## Timeline
- Implement after all LangChain agents are real (non-stub)
- Estimate: Phase 3 of development
