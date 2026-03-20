# Research Workflow — Deep Sentinel Agent Development

**Purpose:** Defines the standard two-phase workflow for every agent in the pipeline.
Phase 1 is research (papers, methods, signals). Phase 2 is implementation (code, tests, validation).
Both phases happen in the same conversation, in order. Never skip Phase 1.

---

## The Core Rule

Every new agent conversation follows this sequence without exception:

```
START CONVERSATION
      │
      ▼
Phase 1 — Research
  ├── Query Research Rabbit for foundational + SOTA papers
  ├── Synthesize findings into a research document
  ├── Define updated output schema based on findings
  └── Confirm methods, libraries, calibration strategy
      │
      ▼
Phase 2 — Implementation
  ├── Write full agent code (STUB_MODE = False)
  ├── Wire into pipeline via LangChain @tool
  ├── Run tests from testing.md
  └── Confirm all 5 pass criteria met
      │
      ▼
END CONVERSATION
Save research doc as: {agent-name}-agent-research.md
Upload to project before closing.
```

Do not split research and implementation across two conversations.
The research context directly shapes implementation decisions — losing it between sessions
creates implementation drift and schema mismatches.

---

## Phase 1 — Research with Research Rabbit

### What is Research Rabbit

Research Rabbit (researchrabbitapp.com) is a free academic paper discovery tool.
It works like a "Spotify for papers" — you give it one seed paper, and it maps the entire
citation network: papers that cite it, papers it cites, and papers by the same authors.
This is the fastest way to find both foundational methods and the latest SOTA for a
specific signal type (frequency, texture, biological, etc.).

### Step-by-Step: How to Use Research Rabbit for Each Agent

**Step 1 — Identify your seed papers.**
Before opening Research Rabbit, identify 1–3 seed papers from the agent's existing spec
in `agents.md`. Every agent already has a "Research Basis" field with starting citations.
Use these as your entry points.

For example, the Frequency Agent seed papers are:
- Durall et al. CVPR 2020 — "Watch Your Up-Convolution"
- Frank et al. ICML 2020 — "Leveraging Frequency Analysis for Deep Fake Image Recognition"

**Step 2 — Add seeds to a new Research Rabbit collection.**
Go to researchrabbitapp.com → create a new collection named after the agent
(e.g., "Deep Sentinel — Frequency Agent"). Add each seed paper by title or DOI.
Research Rabbit will immediately show you the citation graph.

**Step 3 — Explore four directions from each seed.**
For each seed paper, Research Rabbit shows four panels you must check:

- "Similar Work" — papers using the same technique or signal type. These are your peer methods.
  Look for papers with higher accuracy or newer datasets.
- "Later Work" — papers that cited this seed. These are the SOTA extensions built on top of it.
  Sort by year descending — the most recent ones are your implementation targets.
- "Earlier Work" — papers this seed cited. These are the foundational methods. Read these to
  understand why the technique works at a signal/physics level.
- "Authors' Other Work" — the same research group's other papers. Often the best source of
  implementation details and follow-up improvements.

**Step 4 — Filter by relevance using these criteria.**
Not every paper in the graph is worth implementing. Use this filter:

A paper is worth implementing if it meets at least two of these three conditions:
it introduces a new detection signal not already covered by the existing agent methods,
it achieves measurably higher accuracy on FaceForensics++ or Celeb-DF than the current methods,
or it handles a failure mode that the existing methods miss (JPEG compression, diffusion models,
GAN architectures not in FF++).

**Step 5 — Categorize papers into two tiers.**
Tier 1 papers are foundational — they define the core signal the agent is built on.
These must always be implemented. Tier 2 papers are SOTA extensions — they improve accuracy
or generalization. Implement as many as time allows, starting with the highest-impact ones.

**Step 6 — Extract four things from each selected paper.**
For every paper you decide to implement, extract: the exact detection method (algorithm steps),
the dataset used for validation (so you know which test images to use), the reported accuracy
or AUC on a benchmark, and any implementation-level details (block size, wavelet type,
frequency band boundaries, etc.). These go directly into the research document and then
into `config.json` as calibration thresholds.

### Research Rabbit Query Templates by Agent

Use these exact search terms when adding papers to your Research Rabbit collection.
Each line is a search to run. After each result, follow the "Later Work" and "Similar Work"
panels to expand the graph.

**Frequency Agent (L3) — COMPLETED. See `frequency-agent-research.md`.**

**Geometry Agent (L2):**
- "face warping artifact detection deepfake"
- "facial landmark asymmetry GAN detection"
- "3D morphable model face authenticity"
- "dlib 68 landmark deepfake"
- Seed: Li et al. 2018 — "Exposing DeepFake Videos By Detecting Face Warping Artifacts"

**Texture Agent (L4):**
- "local binary pattern face manipulation detection"
- "skin texture inconsistency deepfake"
- "earth mover distance face forgery"
- "Gabor filter deepfake texture analysis"
- Seed: Rossler et al. 2019 — "FaceForensics++: Learning to Detect Manipulated Facial Images"

**Biological Agent (L5):**
- "remote photoplethysmography deepfake detection"
- "rPPG liveness detection synthetic face"
- "corneal reflection GAN image"
- "biological signal face authenticity"
- Seed 1: Ciftci et al. 2020 — "FakeCatcher: Detection of Synthetic Portrait Videos"
- Seed 2: Mittal et al. 2020 — "Detecting Deepfake Videos from Appearance and Behavior"

**VLM Explainability Agent (L6):**
- "vision language model deepfake detection"
- "Grad-CAM face forgery localization"
- "BLIP-2 image authenticity"
- "explainable deepfake detection attention"
- Seed 1: Li et al. 2023 — "BLIP-2: Bootstrapping Language-Image Pre-training"
- Seed 2: Selvaraju et al. 2017 — "Grad-CAM: Visual Explanations from Deep Networks"

**Metadata Forensics Agent (L9):**
- "EXIF metadata image forgery detection"
- "error level analysis deepfake"
- "PRNU camera fingerprint synthetic image"
- "image provenance forensics GAN"
- Seed: Farid 2009 — "Image Forgery Detection" (IEEE Signal Processing Magazine)

**Preprocessing Agent (L1):**
- "face detection landmark extraction deepfake pipeline"
- "MediaPipe FaceMesh accuracy comparison"
- "face alignment preprocessing forgery detection"
- Seed: Lugaresi et al. 2019 — "MediaPipe: A Framework for Building Perception Pipelines"

**Fusion Agent (L7):**
- "ensemble fusion deepfake detection score aggregation"
- "Bayesian fusion multimodal forgery detection"
- "calibrated confidence deepfake classifier"
- Seed: Rossler et al. 2019 — "FaceForensics++" (ensemble baselines section)

---

## Phase 1 Output — What to Produce Before Coding

Before writing a single line of implementation code, the research phase must produce
a complete research document. This document is saved as `{agent-name}-agent-research.md`
and uploaded to the project. The frequency agent's document (`frequency-agent-research.md`)
is the reference example of what a complete research document looks like.

Every research document must contain all of the following sections:

**Section 1 — The Core Insight.** One to three paragraphs explaining why this detection
signal works at a physics or signal-processing level. If you cannot explain why the signal
exists in fake images and not in real ones, the implementation will be fragile.

**Section 2 — Research Landscape.** All selected papers organized into Tier 1 (must implement)
and Tier 2 (SOTA extensions). For each paper: venue, key finding, method summary, and
specifically what to borrow for the implementation.

**Section 3 — Algorithm Stack.** The exact set of methods the agent will implement, shown
as a diagram or list, with the sub-score weighting for the final `anomaly_score`.

**Section 4 onwards — Per-Method Deep Dive.** One section per detection method with
the step-by-step algorithm, complete Python code with comments, and expected output
values from the reference report (DFA-2025-TC-00471) where available.

**Calibration Strategy section.** The expected score ranges for real vs fake images,
per sub-score, with the source paper. These become the normalization thresholds in `config.json`.

**Limitations section.** Known failure modes and how the implementation mitigates them.
This prevents surprises during testing.

**Test Specification section.** The exact assert statements that constitute passing.
These are copied into `testing.md` when the research document is complete.

**Updated Output Schema section.** The final JSON schema for the agent, which may differ
from the placeholder in `agents.md` if research revealed additional useful signals.
Always update `agents.md` and `SKILL.md` with the final schema before coding.

**Citations section.** Full citations for every paper used, including DOI or arXiv ID.

---

## Phase 2 — Implementation Workflow

Once the research document is complete and reviewed, move to implementation in the same
conversation without closing or starting a new one.

**Step 1 — Read `config.json` and `error-handling.md` before writing any code.**
The agent must read its calibration thresholds from `config.json`, not hardcode them.
The `safe_run` wrapper from `error-handling.md` is what the orchestrator uses — the agent's
`run()` function itself should not catch exceptions; the wrapper handles that.

**Step 2 — Write the full implementation with `STUB_MODE = False`.**
Do not write a stub version at this stage. The research is complete. Write the real
implementation immediately. The stub pattern is only for the initial scaffold phase.

**Step 3 — Structure the file exactly as the code generation rules specify.**
Every agent file must have: `STUB_MODE`, `STUB_OUTPUT`, `SCHEMA`, `validate_output()`,
and `run()`. The `run()` function calls private helper functions named `_run_{method}_analysis()`
for each sub-method. Keep `run()` itself under 30 lines — it should only orchestrate,
not compute.

**Step 4 — Run tests from `testing.md` immediately after writing the code.**
Do not move on without testing. The 5 global pass criteria must all pass before the
agent is considered complete.

**Step 5 — Update the Status column in `SKILL.md`.**
Change the agent's status from `STUB` to `IMPLEMENTED` in the agent catalog table.
This is how the next conversation knows what state the pipeline is in.

**Step 6 — Save and upload before closing the conversation.**
The research document (`{agent-name}-agent-research.md`) must be uploaded to the project.
The updated `SKILL.md` must be uploaded. The implemented agent file is in your codebase.

---

## Conversation Opening Template

When starting a new conversation for a new agent, open with this exact message
(replace the bracketed values for each agent):

```
We are building the Deep Sentinel deepfake forensics pipeline.

Current pipeline status:
- [Agent name]: IMPLEMENTED / STUB  ← list all agents with their current status
- Active target: [Agent name] (L[layer number])

Today's workflow:
Phase 1 — Research the [agent name] agent. The signal we are detecting is [one sentence
description of what the agent looks for]. I have already queried Research Rabbit using
the following seeds: [list seed papers]. The key papers I found are: [list 3-5 papers].

Phase 2 — After research is synthesized, implement agents/[agent_name]_agent.py.

Please start by reading agents.md and config.json, then begin the research synthesis.
```

This message gives the new conversation everything it needs: pipeline state, target agent,
signal description, pre-gathered papers, and the two-phase plan. The AI does not need to
ask clarifying questions and can proceed directly to research synthesis.

---

## Why This Workflow Exists

The research-first discipline exists because implementation decisions are downstream
of research decisions. The choice of FFT band boundaries, DCT block size, wavelet type,
score normalization constants — all of these come from the papers, not from intuition.
An agent written without reading the papers will use arbitrary constants that produce
scores outside the expected calibration ranges, causing the fusion agent to produce
inaccurate final verdicts.

The single-conversation rule exists because research context is lossy across conversation
boundaries. When research and implementation happen in the same conversation, the AI
has full access to the reasoning behind every threshold and algorithm choice. When they
are split, the implementation conversation starts cold and fills gaps with guesses.

The Research Rabbit workflow exists because academic citation graphs are dense and non-obvious.
The foundational papers for each signal type were written years apart, across different
venues, and by different groups who were not always aware of each other's work. Research Rabbit
makes the full graph visible in minutes rather than requiring hours of manual literature search.
