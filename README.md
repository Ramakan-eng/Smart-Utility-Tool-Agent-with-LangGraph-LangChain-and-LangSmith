# 🤖 Smart Utility Tool Agent

A conversational utility assistant built with **Streamlit**, **LangChain**, **LangGraph**, and an OpenAI-compatible chat model. It decides whether a question can be answered directly or requires a specialised tool, executes that tool through a controlled LangGraph workflow, then returns a concise natural-language response.

The agent supports arithmetic, UK date/time, selected unit conversions, text statistics, and lookups from a bundled 12-country dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Tooling-1C3C3C)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-black)
![OpenAI](https://img.shields.io/badge/OpenAI-Compatible%20LLM-412991?logo=openai&logoColor=white)
![Temperature](https://img.shields.io/badge/temperature-0-blue)

---

## Table of contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Tool-routing decision logic](#tool-routing-decision-logic)
- [Request lifecycle](#request-lifecycle)
- [Agent state machine](#agent-state-machine)
- [Project structure](#project-structure)
- [Available tools](#available-tools)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Run the web app](#run-the-web-app)
- [Try it](#try-it)
- [Terminal checks](#terminal-checks)
- [How routing works](#how-routing-works)
- [Configuration and observability](#configuration-and-observability)
- [Notebook](#notebook)
- [Dependencies](#dependencies)
- [Current scope and limitations](#current-scope-and-limitations)
- [License](#license)

---

## Highlights

- Tool-aware LLM with deterministic temperature (`0`) for reproducible routing decisions
- Conditional LangGraph routing between reasoning and tool execution via `tools_condition`
- Five focused, single-purpose LangChain tools bound to the model with `bind_tools`
- Streamlit chat interface with in-session conversation history and a clear-chat control
- Local JSON-backed country information (no external API dependency for lookups)
- Optional LangSmith tracing via environment variables for observability

---

## Architecture

```mermaid
flowchart TB
    User(["User"]):::actor

    subgraph PRES["PRESENTATION - main.py"]
        UI["Streamlit Chat UI"]:::iface
        Hist[("Session State\nchat history")]:::store
    end

    subgraph ORCH["ORCHESTRATION - LangGraph StateGraph (agents.py)"]
        GraphStart(("START")):::terminal
        Agent["Agent Node\ninvokes LLM with system prompt"]:::flow
        Router{"tools_condition"}:::decision
        ToolNode["ToolNode\nexecutes requested tool"]:::flow
        GraphEnd(("END")):::terminal

        GraphStart --> Agent
        Agent --> Router
        Router -->|"tool call requested"| ToolNode
        Router -->|"no tool call"| GraphEnd
        ToolNode -->|"tool result appended\nto message state"| Agent
    end

    subgraph MDL["INTELLIGENCE - llm.py"]
        Prompt["System Prompt\nrouting rules"]:::model
        LLM["ChatOpenAI\nOPENAI_MODEL, temperature = 0"]:::model
    end

    subgraph TOOLS["TOOL LAYER - tools.py"]
        direction LR
        Calc["Calculator"]:::tool
        Clock["UK Date / Time"]:::tool
        Conv["Unit Converter"]:::tool
        Text["Text Counter"]:::tool
        Country["Country Lookup"]:::tool
    end

    Data[("data/countries.json\n12 records")]:::store

    UI <--> Hist
    Prompt --> LLM
    Country <--> Data

    User --> UI
    UI -->|"HumanMessage"| GraphStart
    GraphEnd -->|"final answer"| UI
    ToolNode --> Calc
    ToolNode --> Clock
    ToolNode --> Conv
    ToolNode --> Text
    ToolNode --> Country
    Agent -.->|"builds messages"| Prompt
    LLM -.->|"bound via bind_tools"| Agent

    classDef actor fill:#0f172a,stroke:#38bdf8,color:#e0f2fe,stroke-width:2px;
    classDef iface fill:#0c4a6e,stroke:#38bdf8,color:#e0f2fe,stroke-width:1.5px;
    classDef flow fill:#3730a3,stroke:#a5b4fc,color:#eef2ff,stroke-width:1.5px;
    classDef decision fill:#581c87,stroke:#d8b4fe,color:#faf5ff,stroke-width:2px;
    classDef terminal fill:#1e3a8a,stroke:#93c5fd,color:#eff6ff,stroke-width:2px;
    classDef model fill:#7c2d12,stroke:#fdba74,color:#fff7ed,stroke-width:1.5px;
    classDef tool fill:#14532d,stroke:#86efac,color:#f0fdf4,stroke-width:1.5px;
    classDef store fill:#713f12,stroke:#fde047,color:#fefce8,stroke-width:1.5px;
```

**Shape language:** circle = graph `START`/`END` anchor, diamond = routing decision, cylinder = persisted state or data, plain rectangle = a processing node or tool. Every LLM response — the first pass or one after a tool result — is checked by the same `tools_condition` diamond: a tool call routes to `ToolNode`, which appends a `ToolMessage` and loops back into `agent` for another turn; only when a pass produces *no* tool call does `tools_condition` route to `END`, which is the only edge back out to the Streamlit UI.

---

## Tool-routing decision logic

The system prompt in [`agents.py`](agents.py) enforces deterministic rules rather than leaving tool choice purely to the model's judgement. This is what `tools_condition` and the prompt effectively implement:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryTextColor': '#f8fafc', 'lineColor': '#94a3b8'}}}%%
flowchart LR
    Q(["❓ Incoming question"]) --> D1{"Needs\nmath?"}
    D1 -- yes --> CALC["🧮 calculator\n(add / subtract / multiply\n/ divide / percentage)"]
    D1 -- no --> D2{"Needs UK\ndate/time?"}
    D2 -- yes --> CLOCK["📅 current_datetime"]
    D2 -- no --> D3{"Needs unit\nconversion?"}
    D3 -- yes --> CONVERT["📏 convert_units"]
    D3 -- no --> D4{"Needs word/char\n/sentence count?"}
    D4 -- yes --> TEXT["📝 text_counter"]
    D4 -- no --> D5{"Asks about a\nknown country?"}
    D5 -- yes --> COUNTRY["🌍 lookup_country_information"]
    D5 -- no --> DIRECT["💬 Answer directly\n(no tool call)"]

    classDef q fill:#172554,stroke:#60a5fa,color:#eff6ff;
    classDef d fill:#4c1d95,stroke:#c4b5fd,color:#f5f3ff;
    classDef t fill:#365314,stroke:#bef264,color:#f7fee7;
    classDef direct fill:#9a3412,stroke:#fdba74,color:#fff7ed;

    class Q q;
    class D1,D2,D3,D4,D5 d;
    class CALC,CLOCK,CONVERT,TEXT,COUNTRY t;
    class DIRECT direct;
```

Percentage questions receive extra prompt guidance: `18% of 74500` maps to `operation="percentage", a=18, b=74500`, and compound phrasing like "add 12% to 50000" instructs the model to compute the percentage first, then add it to the base value.

---

## Request lifecycle

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'actorBkg': '#1e293b', 'actorBorder': '#64748b', 'actorTextColor': '#e2e8f0', 'signalColor': '#cbd5e1', 'signalTextColor': '#e2e8f0', 'noteBkgColor': '#7c2d12', 'noteTextColor': '#fff7ed', 'noteBorderColor': '#fdba74', 'activationBkgColor': '#334155', 'activationBorderColor': '#94a3b8', 'fontFamily': 'Segoe UI, sans-serif'}}}%%
sequenceDiagram
    autonumber
    actor User
    participant UI as Streamlit App
    participant Agent as Agent Node
    participant LLM as ChatOpenAI (temp=0)
    participant Tools as ToolNode

    User->>+UI: Submit question
    UI->>+Agent: invoke({messages: [HumanMessage]})
    Note over Agent,LLM: System prompt prepended to conversation state
    Agent->>+LLM: SystemMessage + MessagesState
    LLM-->>-Agent: AIMessage

    alt Tool call present
        rect rgba(74, 222, 128, 0.15)
            Agent->>+Tools: route via tools_condition
            Tools-->>-Agent: ToolMessage (result)
            Agent->>+LLM: Updated MessagesState
            LLM-->>-Agent: Final AIMessage
        end
    else No tool required
        Note right of Agent: Model answers directly — no tool call
    end

    Agent-->>-UI: Final message content
    UI-->>-User: Render answer · append to chat history
```

The green band marks the sub-loop that only runs when the model requests a tool — `agent → tools → agent` — before the LangGraph run ever returns control to Streamlit. Activation bars show exactly which component is "on the stack" at each step.

---

## Agent state machine

```mermaid
stateDiagram-v2
    [*] --> Reasoning: user submits prompt
    Reasoning --> Reasoning: no tool needed, direct answer drafted
    Reasoning --> ExecutingTool: AIMessage contains a tool call
    ExecutingTool --> Reasoning: ToolMessage appended to state
    Reasoning --> Done: tools_condition routes to END
    Done --> [*]: answer rendered in Streamlit

    note right of ExecutingTool
        ToolNode dispatches to exactly
        one of: calculator, current_datetime,
        convert_units, text_counter,
        lookup_country_information
    end note
```

---

## Project structure

```text
smart_utility_tool_agent/
├── agents.py              # System prompt, agent node, and compiled LangGraph workflow
├── tools.py               # Five LangChain tool implementations
├── llm.py                 # Environment-driven ChatOpenAI configuration
├── main.py                # Streamlit chat interface
├── test_agent.py          # Manual terminal query harness
├── new.ipynb              # Tutorial / exploratory walkthrough of the agent
├── requirements.txt       # Python dependencies
├── .env.example           # Safe environment-variable template
└── data/
    └── countries.json     # Bundled country lookup dataset (12 records)
```

`__pycache__/`, `.venv/`, and `.env` are local/generated and are excluded from version control via [`.gitignore`](.gitignore).

---

## Available tools

| Tool | Purpose | Supported operations / data |
| --- | --- | --- |
| `calculator` | Performs basic arithmetic | `add`, `subtract`, `multiply`, `divide`, `percentage` |
| `current_datetime` | Returns UK date and time | `Europe/London` timezone; formatted as `DD Month YYYY, HH:MM:SS AM/PM` |
| `convert_units` | Converts common measurements | kilometers↔miles, meters↔feet, kilograms↔pounds, Celsius↔Fahrenheit |
| `text_counter` | Analyses supplied text | words, sentences, characters, characters without spaces |
| `lookup_country_information` | Finds a country in the local dataset | name, capital, currency, population (millions), continent |

The country dataset ([`data/countries.json`](data/countries.json)) currently contains 12 records: Australia, Brazil, Canada, China, France, Germany, India, Japan, Singapore, the United Arab Emirates, the United Kingdom, and the United States. Lookups are case-insensitive exact-name matches.

---

## Prerequisites

- Python 3.10+ (the project uses the standard-library `zoneinfo` module)
- An OpenAI API key
- An OpenAI-compatible model available to your key

---

## Setup

1. Clone or download the project and enter its directory.

2. Create and activate a virtual environment.

   **Windows PowerShell**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and fill in your credentials.

   **Windows PowerShell**

   ```powershell
   Copy-Item .env.example .env
   ```

   **macOS / Linux**

   ```bash
   cp .env.example .env
   ```

5. Configure the required values in `.env`:

   ```dotenv
   OPENAI_API_KEY=sk-your-openai-key-here
   OPENAI_MODEL=gpt-4o-mini
   ```

   The template also includes optional LangSmith tracing variables. Keep `.env` private; it is already ignored by Git.

---

## Run the web app

```bash
streamlit run main.py
```

Streamlit will print a local URL, normally `http://localhost:8501`. The interface provides a sidebar with tool descriptions and example prompts, a clear-conversation button, and a chat input area.

---

## Try it

Examples of requests designed for the available tools:

```text
Calculate 18% VAT on £74,500
Convert 15 kilometers into miles
What is the current date and time in the UK?
Count the words in: LangGraph makes agent workflows explicit.
What are the capital and currency of Japan?
```

Questions that do not need one of the five tools are answered directly by the model.

---

## Terminal checks

`test_agent.py` is an executable manual test harness rather than a pytest suite. It submits representative prompts for every tool category plus a direct conversational prompt.

```bash
python test_agent.py
```

The script requires the same `.env` configuration and network access as the Streamlit app because it invokes the configured LLM.

For a local syntax-only validation that does not call the API:

```bash
python -m py_compile agents.py tools.py llm.py main.py test_agent.py
```

---

## How routing works

[`agents.py`](agents.py) binds all registered tools to the configured LLM with `llm.bind_tools(tools)`. For each request, the `agent` node prepends its system prompt and invokes the model. LangGraph's `tools_condition` inspects the resulting message:

- A normal answer ends the graph and is returned to the UI.
- A tool call is sent to `ToolNode`, which runs the matching Python function from [`tools.py`](tools.py).
- The tool result is appended to `MessagesState` as a `ToolMessage`, then control returns to the `agent` node so the LLM can phrase a clear final response.

This loop allows the model to choose a tool while keeping execution inside application-defined functions and data sources — the LLM never runs arbitrary code, only the five whitelisted tools.

---

## Configuration and observability

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | Credential used by `ChatOpenAI` |
| `OPENAI_MODEL` | Yes | Model identifier passed to `ChatOpenAI` |
| `LANGCHAIN_TRACING` | No | Enables LangSmith tracing when configured |
| `LANGCHAIN_API_KEY` | No | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |
| `LANGCHAIN_ENDPOINT` | No | Optional LangSmith endpoint override |

---

## Notebook

[`new.ipynb`](new.ipynb) is a step-by-step learning notebook covering package installation, tool definitions, model binding, the graph, sample invocations, message inspection, and an earlier textual architecture sketch. The runnable application code lives in the Python modules described above.

---

## Dependencies

The project declares these direct dependencies in [`requirements.txt`](requirements.txt):

- `langchain`
- `langchain-core`
- `langchain-openai`
- `langgraph`
- `python-dotenv`
- `streamlit`

---

## Current scope and limitations

- Country information is limited to exact case-insensitive names in the bundled 12-record dataset.
- Unit conversion intentionally supports the specific pairs listed above; unsupported pairs return a clear message rather than an error.
- The UI sends one new `HumanMessage` per invocation. It preserves rendered chat history in Streamlit session state, while each agent invocation begins with the latest prompt (the LangGraph state is not carried across turns).
- External answers depend on the selected model, API credentials, and network availability.

---

## License

No license file is currently included in this repository. Add one before distributing or reusing the project under specific terms.
