# MASTER PROMPT: Codebase Graphification Framework
## Exhaustive Flow Mapping and Context Engine for AI Agents

---

> **USAGE INSTRUCTION:** Feed this entire document as the system/user prompt to a capable LLM (Claude Sonnet/Opus, GPT-4o, Gemini Ultra) or use it as the specification document for a development team. Every section maps to a concrete, buildable module. Nothing here is aspirational — everything must be implemented.

---

## ═══════════════════════════════════════════════════════
## SECTION 0 — MISSION STATEMENT AND PRIME DIRECTIVE
## ═══════════════════════════════════════════════════════

You are tasked with architecting and implementing **CodeGraphEngine (CGE)** — a production-grade, open-source Python framework that transforms one or more raw source code repositories into a unified, queryable, multi-dimensional knowledge graph. This graph must expose every deterministic flow — control flow, data flow, call chains, business logic, dependency topology, and intent — as first-class citizens queryable by AI agents via a Model Context Protocol (MCP) server.

The framework has **three primary output contracts** that must all be satisfied simultaneously:

1. **GRAPH CONTRACT:** Produce an interactive, clickable, browser-renderable HTML graph showing full upstream/downstream relationships across correlated multi-repo codebases with end-to-end flow tracing.
2. **TEST CONTRACT:** Produce exhaustive Cucumber/Gherkin `.feature` files covering every scenario, input variant, edge case, and boundary condition detectable from the graph.
3. **AGENT CONTRACT:** Serve the entire graph as a local MCP server so AI agents (Claude Code, Cursor, Gemini CLI, Aider, etc.) can query flow paths, blast radii, and taint vectors without reading raw source code.

---

## ═══════════════════════════════════════════════════════
## SECTION 1 — ARCHITECTURAL OVERVIEW
## ═══════════════════════════════════════════════════════

### 1.1 — System Architecture Diagram (Textual)

```
┌────────────────────────────────────────────────────────────────────┐
│                        CodeGraphEngine (CGE)                       │
│                                                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐ │
│  │  INGESTION   │   │   ANALYSIS   │   │      GRAPH STORE       │ │
│  │   LAYER      │──▶│    LAYER     │──▶│  (SQLite / Neo4j /     │ │
│  │              │   │              │   │   NetworkX in-memory)  │ │
│  │ Multi-repo   │   │ AST + CFG +  │   │                        │ │
│  │ Git aware    │   │ PDG + DFG +  │   │  Unified schema across │ │
│  │ Polyglot     │   │ CPG + iCPG   │   │  all repos             │ │
│  └──────────────┘   └──────────────┘   └───────────┬────────────┘ │
│                                                     │              │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────▼────────────┐ │
│  │  CUCUMBER    │   │  INTERACTIVE │   │     MCP SERVER         │ │
│  │  TEST GEN    │◀──│  GRAPH UI    │◀──│  (stdio / HTTP SSE)    │ │
│  │              │   │  (Pyvis /    │   │                        │ │
│  │  .feature    │   │  D3.js /     │   │  30+ Tool endpoints    │ │
│  │  files       │   │  Sigma.js)   │   │  for AI agents         │ │
│  └──────────────┘   └──────────────┘   └────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

### 1.2 — Technology Stack

| Layer | Primary Library | Fallback / Alternative | Purpose |
|---|---|---|---|
| CLI entry | `click` | `typer` | Command-line interface |
| AST parsing (Python) | `ast` (stdlib) + `tree-sitter-python` | `libcst` | Syntax tree extraction |
| AST parsing (JS/TS) | `tree-sitter-javascript` / `tree-sitter-typescript` | `esprima` via subprocess | JS/TS parsing |
| AST parsing (Java) | `tree-sitter-java` | `javalang` | Java parsing |
| AST parsing (Go) | `tree-sitter-go` | subprocess `go ast` | Go parsing |
| AST parsing (C/C++) | `tree-sitter-c` / `tree-sitter-cpp` | `libclang` via `clang` | C/C++ parsing |
| Multi-language unified | `tree-sitter` (Python bindings) | per-language fallbacks | Universal parser |
| Graph engine (local) | `networkx` | `igraph` | In-memory graph computation |
| Graph storage (embedded) | `sqlite3` (stdlib) | `tinydb` | Persistent local storage |
| Graph storage (advanced) | `neo4j` driver | `arangodb` HTTP client | Enterprise graph DB |
| Community detection | `python-louvain` / `leidenalg` | `networkx` modularity | Clustering algorithms |
| Interactive visualization | `pyvis` + `jinja2` | `d3.js` (embedded template) | Clickable HTML graphs |
| Advanced visualization | `sigma.js` (embedded) | `vis-network` | Large graph rendering |
| MCP server | `mcp` (official Python SDK) | `fastapi` SSE fallback | Agent protocol interface |
| Cucumber test gen | Custom + `jinja2` templates | `behave` schema | Gherkin .feature files |
| LLM calls (optional) | `anthropic` SDK | `openai` SDK | Semantic enrichment |
| Git integration | `gitpython` | subprocess `git` | Repo metadata, blame, history |
| Config management | `pydantic-settings` | `dynaconf` | Schema-validated config |
| Dependency resolution | `pipdeptree` (Python) | `npm list --json` (Node) | Package dep graphs |
| Taint analysis | Custom graph traversal | Integrate `joern` via subprocess | Source-to-sink paths |
| Concurrency | `concurrent.futures` | `asyncio` | Parallel repo ingestion |
| Serialization | `orjson` | `ujson` | Fast JSON I/O |

### 1.3 — Directory Layout (Generated Project Structure)

```
codegraphengine/
├── cge/
│   ├── __init__.py
│   ├── cli.py                        # Click CLI entry points
│   ├── config.py                     # Pydantic settings & schema
│   │
│   ├── ingestion/                    # Layer 1: Pull & normalize repos
│   │   ├── __init__.py
│   │   ├── repo_loader.py            # Git clone, local path, monorepo
│   │   ├── file_walker.py            # Language-aware file discovery
│   │   ├── language_detector.py      # Per-file language classification
│   │   └── multi_repo_correlator.py  # Cross-repo linking logic
│   │
│   ├── parsers/                      # Layer 2: Per-language AST parsers
│   │   ├── __init__.py
│   │   ├── base_parser.py            # Abstract interface
│   │   ├── python_parser.py
│   │   ├── javascript_parser.py
│   │   ├── typescript_parser.py
│   │   ├── java_parser.py
│   │   ├── go_parser.py
│   │   ├── csharp_parser.py
│   │   ├── cpp_parser.py
│   │   ├── ruby_parser.py
│   │   ├── rust_parser.py
│   │   └── generic_treesitter_parser.py  # Fallback for all others
│   │
│   ├── analysis/                     # Layer 3: Graph construction engines
│   │   ├── __init__.py
│   │   ├── ast_analyzer.py           # AST → node/edge extraction
│   │   ├── cfg_builder.py            # Control Flow Graph per function
│   │   ├── pdg_builder.py            # Program Dependence Graph
│   │   ├── dfg_builder.py            # Data Flow Graph (mutations)
│   │   ├── call_graph_builder.py     # Inter-procedural call chains
│   │   ├── import_graph_builder.py   # Module/package dependency graph
│   │   ├── taint_analyzer.py         # Source-to-sink taint tracking
│   │   ├── topology_classifier.py    # Node role classification
│   │   ├── blast_radius_calculator.py# Impact simulation engine
│   │   ├── intent_extractor.py       # iCPG: business logic/intent layer
│   │   ├── community_detector.py     # Leiden / Louvain clustering
│   │   └── cross_repo_linker.py      # Upstream/downstream repo edges
│   │
│   ├── graph/                        # Layer 4: Graph storage & querying
│   │   ├── __init__.py
│   │   ├── schema.py                 # Node/Edge dataclass definitions
│   │   ├── store_sqlite.py           # SQLite-backed persistence
│   │   ├── store_neo4j.py            # Neo4j driver adapter
│   │   ├── store_networkx.py         # In-memory NetworkX adapter
│   │   ├── query_engine.py           # Unified query interface
│   │   ├── path_finder.py            # Shortest path, BFS, DFS
│   │   └── graph_merger.py           # Multi-repo graph union
│   │
│   ├── visualization/                # Layer 5: Interactive graph rendering
│   │   ├── __init__.py
│   │   ├── html_renderer.py          # Pyvis → clickable HTML
│   │   ├── sigma_renderer.py         # Sigma.js large-scale graphs
│   │   ├── d3_renderer.py            # D3.js force-directed graphs
│   │   ├── multi_repo_view.py        # Cross-repo upstream/downstream
│   │   ├── flow_highlighter.py       # Highlight specific paths
│   │   └── templates/
│   │       ├── graph_base.html.j2
│   │       ├── sigma_base.html.j2
│   │       └── flow_detail.html.j2
│   │
│   ├── mcp_server/                   # Layer 6: MCP tool endpoints
│   │   ├── __init__.py
│   │   ├── server.py                 # MCP server bootstrap
│   │   ├── tools/
│   │   │   ├── graph_tools.py        # Query graph, neighbors, paths
│   │   │   ├── flow_tools.py         # CFG, DFG, call chain tools
│   │   │   ├── taint_tools.py        # Source-to-sink queries
│   │   │   ├── blast_radius_tools.py # Impact analysis tools
│   │   │   ├── topology_tools.py     # Node role queries
│   │   │   ├── cross_repo_tools.py   # Upstream/downstream queries
│   │   │   └── intent_tools.py       # Business logic / iCPG tools
│   │   └── resources/
│   │       └── graph_resources.py    # MCP resource registrations
│   │
│   ├── test_generation/              # Layer 7: Cucumber test generation
│   │   ├── __init__.py
│   │   ├── scenario_extractor.py     # Derive test scenarios from graph
│   │   ├── input_classifier.py       # Input type / boundary analysis
│   │   ├── gherkin_writer.py         # Render .feature files
│   │   ├── step_definition_gen.py    # Stub step_defs in Python/Java/JS
│   │   └── templates/
│   │       ├── feature_base.feature.j2
│   │       ├── scenario_happy_path.j2
│   │       ├── scenario_edge_case.j2
│   │       ├── scenario_error_path.j2
│   │       └── scenario_security.j2
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                 # Structured logging
│       ├── hash_utils.py             # SHA256 incremental diff
│       ├── file_utils.py
│       └── llm_client.py             # Optional LLM enrichment calls
│
├── tests/
│   └── ...                           # CGE's own test suite
├── examples/
│   └── sample_repos/
├── pyproject.toml
├── cge.config.yaml                   # Example config
└── README.md
```

---

## ═══════════════════════════════════════════════════════
## SECTION 2 — INGESTION LAYER (Exhaustive Specification)
## ═══════════════════════════════════════════════════════

### 2.1 — Repository Loader (`ingestion/repo_loader.py`)

Implement `RepoLoader` with the following input modes. All modes must normalize into a canonical `RepoManifest` dataclass before passing downstream:

**Input Modes:**
- `--repo /path/to/local/dir` — local directory, supports monorepos
- `--repo https://github.com/org/repo` — HTTPS clone via `gitpython`
- `--repo git@github.com:org/repo.git` — SSH clone
- `--repos repos.yaml` — YAML manifest listing N repos with optional `role:` field (`upstream`, `downstream`, `shared-lib`, `schema`, `config`)
- `--repo . --branch feature/xyz` — specific branch
- `--repo . --commit abc123` — pin to specific commit (for reproducibility)

**`RepoManifest` Schema:**
```python
@dataclass
class RepoManifest:
    repo_id: str               # Unique stable hash of path+commit
    local_path: Path           # Resolved local disk path
    remote_url: Optional[str]  # Original URL if cloned
    branch: str
    commit_sha: str
    role: str                  # upstream | downstream | shared-lib | schema | config | standalone
    language_hints: list[str]  # Pre-detected primary languages
    git_log: list[GitCommit]   # Last N commits for temporal edges
    created_at: datetime
```

**Multi-Repo Correlation Rules:**
- When multiple repos are loaded, CGE must auto-detect shared interfaces by:
  1. Matching exported symbols (function names, class names, REST route strings) across repos
  2. Matching `package.json` / `pom.xml` / `requirements.txt` dependency declarations where one repo depends on another
  3. Matching OpenAPI/Swagger endpoint definitions to HTTP client call sites in other repos
  4. Matching gRPC proto definitions to generated stub imports
  5. Matching database table names referenced across repos (ORM models vs raw SQL strings)
- All cross-repo matches become `CROSS_REPO_EDGE` typed edges in the unified graph

### 2.2 — File Walker (`ingestion/file_walker.py`)

Must recursively discover all parseable files, respecting:
- `.gitignore` rules (use `gitpython`'s ignore logic)
- Custom `--exclude` glob patterns passed via CLI or config
- Binary file detection (skip binaries silently, log count)
- Symlink resolution (with cycle detection)
- File size limits (configurable, default skip files > 5MB)

**Language detection priority:**
1. File extension mapping (`.py`, `.ts`, `.java`, `.go`, `.rb`, `.rs`, `.cs`, `.cpp`, `.c`, `.kt`, `.swift`)
2. Shebang line (`#!/usr/bin/env python3`)
3. Content heuristics (first 512 bytes)
4. Tree-sitter `detect_language` fallback

**Special file types that must be parsed (not skipped):**
- `*.sql` — extract table names, stored procedures, views → `DB_SCHEMA` nodes
- `*.proto` — extract gRPC service/method definitions → `RPC_ENDPOINT` nodes
- `*.yaml` / `*.yml` — extract OpenAPI specs, Kubernetes manifests, GitHub Actions workflows
- `*.json` — extract `package.json` deps, `tsconfig.json` path aliases
- `*.toml` — `pyproject.toml`, `Cargo.toml`
- `Dockerfile` / `docker-compose.yml` — extract service topology → `INFRA` nodes
- `*.tf` (Terraform) — extract resource definitions → `INFRA` nodes
- `*.md` — extract intent blocks (`<!-- intent: ... -->` or `// intent:` comments)
- `*.feature` (existing Cucumber) — parse to detect already-tested scenarios

---

## ═══════════════════════════════════════════════════════
## SECTION 3 — ANALYSIS LAYER (Core Graph Construction)
## ═══════════════════════════════════════════════════════

### 3.1 — Universal Node Schema (`graph/schema.py`)

Every entity in the graph is a `GraphNode`. Every relationship is a `GraphEdge`. These are the master schemas — never deviate from them.

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any

class NodeType(str, Enum):
    # Code structure
    REPOSITORY        = "REPOSITORY"
    FILE              = "FILE"
    MODULE            = "MODULE"
    PACKAGE           = "PACKAGE"
    CLASS             = "CLASS"
    INTERFACE         = "INTERFACE"
    TRAIT             = "TRAIT"
    ENUM              = "ENUM"
    FUNCTION          = "FUNCTION"
    METHOD            = "METHOD"
    CONSTRUCTOR       = "CONSTRUCTOR"
    LAMBDA            = "LAMBDA"
    VARIABLE          = "VARIABLE"
    PARAMETER         = "PARAMETER"
    CONSTANT          = "CONSTANT"
    DECORATOR         = "DECORATOR"
    ANNOTATION        = "ANNOTATION"

    # Flow constructs
    CFG_ENTRY         = "CFG_ENTRY"
    CFG_EXIT          = "CFG_EXIT"
    CFG_BRANCH        = "CFG_BRANCH"
    CFG_LOOP          = "CFG_LOOP"
    CFG_EXCEPTION     = "CFG_EXCEPTION"
    CFG_RETURN        = "CFG_RETURN"

    # Data entities
    DB_TABLE          = "DB_TABLE"
    DB_COLUMN         = "DB_COLUMN"
    DB_PROCEDURE      = "DB_PROCEDURE"
    DB_VIEW           = "DB_VIEW"
    SCHEMA_MODEL      = "SCHEMA_MODEL"      # ORM models (SQLAlchemy, Prisma, etc.)
    DTO               = "DTO"               # Data Transfer Object
    TYPE_ALIAS        = "TYPE_ALIAS"

    # API / RPC
    API_ENDPOINT      = "API_ENDPOINT"      # REST, GraphQL, gRPC
    API_ROUTE         = "API_ROUTE"
    HTTP_CLIENT_CALL  = "HTTP_CLIENT_CALL"
    RPC_DEFINITION    = "RPC_DEFINITION"
    RPC_CALL          = "RPC_CALL"
    EVENT_EMITTER     = "EVENT_EMITTER"
    EVENT_CONSUMER    = "EVENT_CONSUMER"
    MESSAGE_QUEUE     = "MESSAGE_QUEUE"

    # Security / taint
    TAINT_SOURCE      = "TAINT_SOURCE"
    TAINT_SINK        = "TAINT_SINK"
    SANITIZER         = "SANITIZER"

    # Infrastructure
    SERVICE           = "SERVICE"           # Docker/K8s service
    INFRA_RESOURCE    = "INFRA_RESOURCE"    # Terraform resource
    ENV_VAR           = "ENV_VAR"           # Environment variable reference
    CONFIG_KEY        = "CONFIG_KEY"        # Explicit config key access

    # Intent / business logic (iCPG tiers)
    INTENT            = "INTENT"            # Tier 1: Why
    INVARIANT         = "INVARIANT"         # Tier 2: Contract/constraint
    BUSINESS_RULE     = "BUSINESS_RULE"     # Tier 2: Domain rule

    # Cross-repo
    EXTERNAL_DEP      = "EXTERNAL_DEP"      # Third-party package node
    CROSS_REPO_SYMBOL = "CROSS_REPO_SYMBOL" # Symbol resolved in another repo

class EdgeType(str, Enum):
    # AST relationships
    DEFINES           = "DEFINES"           # FILE defines CLASS/FUNCTION
    CONTAINS          = "CONTAINS"          # CLASS contains METHOD
    INHERITS          = "INHERITS"          # CLASS inherits CLASS
    IMPLEMENTS        = "IMPLEMENTS"        # CLASS implements INTERFACE
    ANNOTATED_BY      = "ANNOTATED_BY"      # METHOD annotated by DECORATOR

    # Call graph
    CALLS             = "CALLS"             # FUNCTION calls FUNCTION
    CALLS_ASYNC       = "CALLS_ASYNC"       # async invocation
    CALLS_CONDITIONAL = "CALLS_CONDITIONAL" # Call inside if branch
    RETURNS_TO        = "RETURNS_TO"        # Return value flows to caller

    # Data flow
    READS             = "READS"             # Function reads VARIABLE
    WRITES            = "WRITES"            # Function writes VARIABLE
    PASSES            = "PASSES"            # Passes value as parameter
    MUTATES           = "MUTATES"           # In-place mutation
    PROPAGATES        = "PROPAGATES"        # Data flows through

    # Control flow
    BRANCHES_TO       = "BRANCHES_TO"       # CFG conditional branch
    LOOPS_BACK        = "LOOPS_BACK"        # CFG loop back-edge
    RAISES            = "RAISES"            # Exception raising
    CATCHES           = "CATCHES"           # Exception handling

    # Module / dependency
    IMPORTS           = "IMPORTS"           # Module imports another
    DEPENDS_ON        = "DEPENDS_ON"        # Package dependency
    EXPORTS           = "EXPORTS"           # Module exports symbol

    # API
    SERVES            = "SERVES"            # SERVICE serves API_ENDPOINT
    CALLS_API         = "CALLS_API"         # HTTP_CLIENT_CALL calls API_ENDPOINT
    PUBLISHES_TO      = "PUBLISHES_TO"      # EVENT_EMITTER publishes to QUEUE
    CONSUMES_FROM     = "CONSUMES_FROM"     # EVENT_CONSUMER consumes from QUEUE

    # Database
    QUERIES           = "QUERIES"           # FUNCTION queries DB_TABLE
    WRITES_TO_DB      = "WRITES_TO_DB"
    MAPS_TO           = "MAPS_TO"           # ORM MODEL maps to DB_TABLE

    # Taint
    TAINT_FLOWS       = "TAINT_FLOWS"       # Tainted data flows from A to B
    SANITIZED_BY      = "SANITIZED_BY"      # Flow sanitized at this node

    # Intent
    IMPLEMENTS_INTENT = "IMPLEMENTS_INTENT" # CODE implements INTENT
    ENFORCES          = "ENFORCES"          # CODE enforces INVARIANT
    VIOLATES          = "VIOLATES"          # CODE violates INVARIANT (detected)

    # Cross-repo
    UPSTREAM_OF       = "UPSTREAM_OF"       # Repo A is upstream of Repo B
    DOWNSTREAM_OF     = "DOWNSTREAM_OF"     # Repo B is downstream of Repo A
    CROSS_REPO_CALLS  = "CROSS_REPO_CALLS"  # Function in Repo A calls Repo B
    CROSS_REPO_IMPORT = "CROSS_REPO_IMPORT"

class NodeRole(str, Enum):
    ENTRY     = "ENTRY"       # High out-degree, zero in-degree (API controllers, CLIs)
    CORE      = "CORE"        # High in+out-degree (critical shared logic)
    UTILITY   = "UTILITY"     # High in-degree, low out-degree (helpers, formatters)
    ADAPTER   = "ADAPTER"     # External SDK/DB interface boundary
    DEAD      = "DEAD"        # Zero in-degree, zero out-degree (unused code)
    LEAF      = "LEAF"        # Terminal execution with no further calls
    BRIDGE    = "BRIDGE"      # Cross-repo connectivity nodes
    GATEWAY   = "GATEWAY"     # Cross-service API gateway nodes

class ConfidenceLevel(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"   # From AST/CPG — 100% accurate
    INFERRED      = "INFERRED"        # From LLM semantic enrichment — probabilistic
    HEURISTIC     = "HEURISTIC"       # From pattern matching — medium confidence

@dataclass
class GraphNode:
    node_id: str                              # Stable SHA256 of repo+file+symbol+line
    repo_id: str
    node_type: NodeType
    name: str
    qualified_name: str                       # e.g., "com.example.ServiceA.processOrder"
    file_path: str
    start_line: int
    end_line: int
    language: str
    role: Optional[NodeRole] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys (all optional): signature, docstring, return_type, parameters,
    # visibility, is_async, is_static, decorators, complexity_score, test_coverage,
    # git_last_modified, git_author, git_churn_count, intent_text, invariants

@dataclass
class GraphEdge:
    edge_id: str                              # SHA256 of source+target+type
    source_id: str
    target_id: str
    edge_type: EdgeType
    confidence: ConfidenceLevel = ConfidenceLevel.DETERMINISTIC
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys: call_site_line, call_site_file, condition_text,
    # data_type, taint_category, cross_repo_match_method
```

### 3.2 — AST Analyzer (`analysis/ast_analyzer.py`)

For every language, extract the following node types without omission:

**Python extraction targets:**
- All `def` and `async def` → `FUNCTION` / `METHOD`
- All `class` definitions including `@dataclass` → `CLASS`
- All `import` and `from...import` → `IMPORTS` edges
- All `__init__` constructors → `CONSTRUCTOR`
- All `@decorator` usages → `DECORATOR` nodes + `ANNOTATED_BY` edges
- All type aliases (`MyType = ...`) → `TYPE_ALIAS`
- All `global` and `nonlocal` variable declarations → `VARIABLE`
- All `raise XxxError` → `RAISES` edge in CFG
- All `except` blocks → `CATCHES` edges
- Lambda expressions → `LAMBDA` nodes
- List/dict/set comprehensions with function calls → inline `CALLS` edges
- `yield` / `yield from` / `async for` → tagged in metadata
- `__all__` exports → `EXPORTS` edges

**JavaScript / TypeScript extras:**
- Arrow functions, function expressions → `LAMBDA` / `FUNCTION`
- `export default` / `export const` → `EXPORTS` edges
- `interface` / `type` → `INTERFACE` / `TYPE_ALIAS`
- `enum` → `ENUM`
- Generic type parameters → stored in metadata
- React component detection: any function returning JSX → tagged `IS_REACT_COMPONENT: true`, `IS_HOOK: true` if name starts with `use`
- Express route decorators: `app.get('/path', handler)` → `API_ROUTE` node
- NestJS `@Controller`, `@Get`, `@Post` annotations → `API_ENDPOINT` nodes
- `fetch()` / `axios.*` / `superagent` calls → `HTTP_CLIENT_CALL` nodes

**Java extras:**
- Spring Boot `@RestController`, `@RequestMapping`, `@GetMapping` → `API_ENDPOINT`
- JPA `@Entity`, `@Table` → `SCHEMA_MODEL` + `MAPS_TO` edge to `DB_TABLE`
- `@Autowired` / dependency injection fields → `DEPENDS_ON` edges
- Interface implementations → `IMPLEMENTS` edges
- Abstract class methods → tagged in metadata

### 3.3 — Control Flow Graph Builder (`analysis/cfg_builder.py`)

For every `FUNCTION` and `METHOD` node, build an **intraprocedural CFG**:

**Required CFG node types to track:**
- Entry point (`CFG_ENTRY`)
- Linear statement blocks (collapsed into single nodes for efficiency)
- `if`/`elif`/`else` branches → `CFG_BRANCH` nodes with condition text stored in metadata
- `for` / `while` loops → `CFG_LOOP` nodes with back-edge (`LOOPS_BACK`) to loop head
- `try` / `except` / `finally` → `CFG_EXCEPTION` nodes with exception type metadata
- `return` statements → `CFG_RETURN` nodes
- `break` / `continue` → edges to loop exit / loop head respectively
- `raise` → `RAISES` edge to nearest enclosing `except` or function exit
- Exit point (`CFG_EXIT`)

**Interprocedural CFG:**
- For every `CALLS` edge where the callee is defined within the same repo (or a linked cross-repo), generate a `CALL_CHAIN` object storing the ordered sequence of function activations
- Build **call chains** using BFS up to a configurable depth (default: 10 hops)
- Store call chains in the graph store as ordered path objects, queryable by MCP tools

### 3.4 — Data Flow Graph Builder (`analysis/dfg_builder.py`)

Track variables through assignment chains within and across function boundaries:

**Intra-function DFG:**
- For every variable assignment (`x = expr`), create a `WRITES` edge from `FUNCTION` to `VARIABLE` node
- For every variable read (`use(x)`), create a `READS` edge
- Track SSA (Static Single Assignment) form where possible: each re-assignment creates a new `VARIABLE` node version, linked by `PROPAGATES` edge
- Track mutable parameter mutation: if a parameter object is mutated, emit `MUTATES` edge

**Inter-function DFG:**
- Argument passing: `foo(x)` → `PASSES` edge from call site to callee's parameter node
- Return value flows: `result = foo()` → `RETURNS_TO` edge from callee's return to caller's variable
- Global variable shared-state tracking: functions that both read and write the same global → emit both `READS` and `WRITES` edges to same `VARIABLE`

### 3.5 — Taint Analyzer (`analysis/taint_analyzer.py`)

Implement a **source-to-sink taint propagation engine**:

**Pre-defined taint source patterns (extensible via config):**
- HTTP request body/params: Flask `request.form`, `request.args`, `request.json`, FastAPI `Body()`, Express `req.body`, `req.query`, `req.params`
- File reads: `open(path).read()`, `fs.readFileSync`, `Files.readAllBytes`
- Database reads: `cursor.fetchall()`, ORM `.query()` results
- Environment variables: `os.environ.get()`, `process.env.*`
- User input: `input()`, `sys.argv`, CLI argument values
- External API responses: `requests.get().json()`, `axios.get().data`

**Pre-defined taint sink patterns:**
- SQL execution: `cursor.execute(query)`, `db.query(sql)`, `EntityManager.createNativeQuery`
- Shell execution: `os.system()`, `subprocess.run(shell=True)`, `exec()`, `eval()`
- HTML rendering: `render_template_string`, `innerHTML =`, `dangerouslySetInnerHTML`
- File write: `open(path, 'w').write(user_input)`, `fs.writeFileSync`
- Network redirect: `redirect(user_controlled_url)`
- Deserialization: `pickle.loads()`, `yaml.load()` without `Loader=SafeLoader`

**Sanitizer detection:**
- Parameterized queries: `cursor.execute(query, (param,))` — marks as sanitized
- HTML escaping: `html.escape()`, `bleach.clean()`, `DOMPurify.sanitize()`
- Input validation libraries: `pydantic` model validation, `joi.validate()`, `javax.validation`
- Type casting with validation functions

**Output:** For each taint path found, emit a `TaintPath` object:
```python
@dataclass
class TaintPath:
    path_id: str
    source_node: GraphNode
    sink_node: GraphNode
    path: list[GraphNode]   # Ordered traversal from source to sink
    vulnerability_class: str  # SQLi, XSS, SSRF, CMDi, PathTraversal, etc.
    is_sanitized: bool
    sanitizer_nodes: list[GraphNode]
    confidence: ConfidenceLevel
    cross_repo: bool          # Does the path cross a repo boundary?
```

### 3.6 — Topology Classifier (`analysis/topology_classifier.py`)

After full graph construction, classify every `FUNCTION`, `METHOD`, `CLASS`, and `MODULE` node:

**Algorithm:**
1. Compute `in_degree` = number of `CALLS`, `IMPORTS`, `DEPENDS_ON` edges pointing TO the node (from within the same graph boundary)
2. Compute `out_degree` = number of outgoing edges
3. Apply classification matrix:

| Role | Condition |
|---|---|
| `ENTRY` | `in_degree == 0` AND `out_degree > 0` AND node is `FUNCTION`/`METHOD`/`API_ROUTE` |
| `CORE` | `in_degree > threshold_high` AND `out_degree > threshold_high` |
| `UTILITY` | `in_degree > threshold_medium` AND `out_degree <= threshold_low` |
| `ADAPTER` | Node imports from `EXTERNAL_DEP` AND has significant internal callers |
| `DEAD` | `in_degree == 0` AND `out_degree == 0` AND NOT `ENTRY` |
| `LEAF` | `out_degree == 0` AND `in_degree > 0` |
| `BRIDGE` | Has `CROSS_REPO_CALLS` or `CROSS_REPO_IMPORT` edges |
| `GATEWAY` | Is `API_ENDPOINT` type that is called by external HTTP clients |

Thresholds are configurable (`threshold_high` default: 5, `threshold_medium` default: 2, `threshold_low` default: 1).

**Blast Radius Calculation:**
- For any given node N, blast radius = all nodes reachable from N via outgoing `CALLS`, `IMPORTS`, `PROPAGATES` edges (transitive closure)
- Express as: `{ "direct": [list], "transitive": [list], "cross_repo": [list], "count": int, "risk_score": float }`
- Risk score formula: `(core_node_count * 3 + entry_node_count * 2 + cross_repo_count * 4) / total_blast_count`

### 3.7 — Intent Extractor (`analysis/intent_extractor.py`)

Implement the three-tier iCPG model:

**Tier 1 — Reason Graph (Intents/Whys):**
- Scan all source files for inline intent comments:
  - Python: `# intent: <text>`, `# why: <text>`
  - JS/TS: `// intent: <text>`, `/** @intent <text> */`
  - Java: `// intent: <text>`, `@Intent("<text>")` annotations
- Scan `README.md`, `ARCHITECTURE.md`, `CLAUDE.md`, `AGENTS.md` for architectural decision records (ADRs)
- Optional LLM enrichment: pass each `CLASS` docstring + method signatures to LLM and ask it to extract a single-sentence intent. Store with `confidence: INFERRED`
- Create `INTENT` nodes and `IMPLEMENTS_INTENT` edges to implementing code nodes

**Tier 2 — Semantic Graph (Contracts/Whats):**
- Extract all database schema definitions → `SCHEMA_MODEL` nodes
- Extract all API endpoint definitions with their request/response types → `API_ENDPOINT` nodes
- Extract all `INVARIANT` annotations or comments: `# invariant: X must always be true`
- Detect common invariant patterns automatically:
  - Auth middleware always called before route handlers → create `INVARIANT` node
  - All DB writes wrapped in transaction decorators
  - All public methods have type annotations (configurable as team rule)
- Create `ENFORCES` and `VIOLATES` edges to code nodes

**Tier 3 — Physical Graph (Code/Hows):**
- This is the standard AST/CFG/DFG layer — already built above
- Link Tier 3 nodes UP to Tier 2 contracts and Tier 1 intents

### 3.8 — Community Detector (`analysis/community_detector.py`)

Apply community detection to identify architectural "neighborhoods":

**Algorithm options (configurable):**
- `leiden` (via `leidenalg` library) — preferred for large graphs
- `louvain` (via `python-louvain`) — faster fallback
- `infomap` (via `infomap` library) — best for directed graphs

**Process:**
1. Build undirected projection of the graph (collapse edge directionality)
2. Apply chosen algorithm, producing a partition map: `{node_id: community_id}`
3. For each community, generate a `COMMUNITY` node (a "god node") with:
   - `member_count`
   - `primary_language`
   - `dominant_node_types`
   - `inferred_purpose` (optional LLM call: "Given these function names, what does this module cluster do?")
   - `internal_edge_density` (ratio of internal edges to possible edges)
4. Create `BELONGS_TO` edges from member nodes to their `COMMUNITY` node
5. Create `COMMUNITY_CONNECTS` edges between communities that have cross-community edges

### 3.9 — Cross-Repo Linker (`analysis/cross_repo_linker.py`)

After all individual repos are graphed, merge them into a unified graph and create cross-repo edges:

**Linking strategies (apply all, rank by confidence):**

1. **Package manifest matching** (DETERMINISTIC):
   - `requirements.txt` / `setup.py` / `pyproject.toml`: if Repo B's package name appears in Repo A's dependencies, create `UPSTREAM_OF` edge from Repo B to Repo A
   - `package.json` dependencies, `pom.xml` dependencies, `go.mod` require statements — same logic

2. **API contract matching** (DETERMINISTIC if OpenAPI spec exists, HEURISTIC otherwise):
   - If Repo A defines an OpenAPI spec with endpoint `/api/v1/orders`, scan Repo B for HTTP client calls to that path
   - Match via exact string, then via regex, then via semantic similarity (LLM optional)
   - Create `CROSS_REPO_CALLS` edge from the HTTP client node in Repo B to the `API_ENDPOINT` node in Repo A

3. **Symbol name matching** (HEURISTIC):
   - If Repo A exports `class OrderService` and Repo B has a `from order_service import OrderService` or JS `require('order-service')`, create `CROSS_REPO_IMPORT` edge
   - Threshold: only match if symbol name AND module path share >70% similarity

4. **Database table matching** (DETERMINISTIC if schema file exists):
   - If Repo A defines `DB_TABLE` node `orders` and Repo B has SQL string `SELECT * FROM orders`, create `CROSS_REPO_CALLS` edge with `edge_type: QUERIES`

5. **Event/message matching** (HEURISTIC):
   - If Repo A publishes to message queue topic `order.created` and Repo B subscribes to same topic name, create `UPSTREAM_OF` + `PUBLISHES_TO` / `CONSUMES_FROM` edges

**Upstream/Downstream Direction Convention:**
- Data producer → `UPSTREAM_OF` → data consumer
- Library/service provider → `UPSTREAM_OF` → library/service consumer
- Database schema owner → `UPSTREAM_OF` → all services reading from it

---

## ═══════════════════════════════════════════════════════
## SECTION 4 — GRAPH STORAGE LAYER
## ═══════════════════════════════════════════════════════

### 4.1 — SQLite Store (`graph/store_sqlite.py`)

**Schema (create on first run):**
```sql
CREATE TABLE IF NOT EXISTS repositories (
    repo_id TEXT PRIMARY KEY,
    path TEXT, url TEXT, branch TEXT, commit_sha TEXT,
    role TEXT, language_hints TEXT, ingested_at TEXT
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    repo_id TEXT,
    node_type TEXT, name TEXT, qualified_name TEXT,
    file_path TEXT, start_line INT, end_line INT,
    language TEXT, role TEXT,
    metadata_json TEXT,
    FOREIGN KEY(repo_id) REFERENCES repositories(repo_id)
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id TEXT PRIMARY KEY,
    source_id TEXT, target_id TEXT,
    edge_type TEXT, confidence TEXT,
    weight REAL, metadata_json TEXT,
    FOREIGN KEY(source_id) REFERENCES nodes(node_id),
    FOREIGN KEY(target_id) REFERENCES nodes(node_id)
);

CREATE TABLE IF NOT EXISTS taint_paths (
    path_id TEXT PRIMARY KEY,
    source_node_id TEXT, sink_node_id TEXT,
    path_json TEXT, vulnerability_class TEXT,
    is_sanitized BOOLEAN, sanitizer_json TEXT,
    confidence TEXT, cross_repo BOOLEAN
);

CREATE TABLE IF NOT EXISTS communities (
    community_id TEXT PRIMARY KEY,
    member_ids_json TEXT, inferred_purpose TEXT,
    internal_edge_density REAL
);

CREATE TABLE IF NOT EXISTS file_hashes (
    file_path TEXT PRIMARY KEY,
    sha256 TEXT, last_indexed TEXT
);

CREATE INDEX idx_nodes_type ON nodes(node_type);
CREATE INDEX idx_nodes_repo ON nodes(repo_id);
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_type ON edges(edge_type);
```

**Incremental indexing:** Before re-parsing any file, compute its SHA256. If it matches the stored hash in `file_hashes`, skip it. Only re-analyze modified files and recompute affected edges.

### 4.2 — Query Engine (`graph/query_engine.py`)

Expose a **unified query interface** that works regardless of whether the backend is SQLite, Neo4j, or in-memory NetworkX:

```python
class GraphQueryEngine:
    def get_node(self, node_id: str) -> GraphNode: ...
    def get_neighbors(self, node_id: str, direction: str = "both", edge_types: list = None, depth: int = 1) -> list[GraphNode]: ...
    def get_call_chain(self, from_node_id: str, to_node_id: str) -> list[GraphNode]: ...
    def get_blast_radius(self, node_id: str, max_depth: int = 5) -> BlastRadiusResult: ...
    def get_taint_paths(self, source_id: str = None, sink_id: str = None, vuln_class: str = None) -> list[TaintPath]: ...
    def get_nodes_by_role(self, role: NodeRole, repo_id: str = None) -> list[GraphNode]: ...
    def get_cross_repo_paths(self, from_repo: str, to_repo: str) -> list[list[GraphNode]]: ...
    def get_dead_code(self, repo_id: str = None) -> list[GraphNode]: ...
    def get_entry_points(self, repo_id: str = None) -> list[GraphNode]: ...
    def get_api_endpoints(self) -> list[GraphNode]: ...
    def get_data_flow(self, variable_node_id: str) -> DataFlowResult: ...
    def get_community(self, community_id: str) -> CommunityResult: ...
    def search_nodes(self, query: str, node_types: list = None, fuzzy: bool = True) -> list[GraphNode]: ...
    def get_shortest_path(self, from_id: str, to_id: str) -> list[GraphNode]: ...
    def get_upstream_repos(self, repo_id: str) -> list[str]: ...
    def get_downstream_repos(self, repo_id: str) -> list[str]: ...
    def get_end_to_end_flow(self, entry_node_id: str) -> EndToEndFlow: ...
```

---

## ═══════════════════════════════════════════════════════
## SECTION 5 — VISUALIZATION LAYER (Clickable Graphs)
## ═══════════════════════════════════════════════════════

### 5.1 — Core Requirements

All visualizations must produce **self-contained single HTML files** that:
- Open in any modern browser with zero external network dependency
- Embed all JS/CSS inline (no CDN calls at runtime)
- Are clickable: clicking a node shows a right-side detail panel with the node's full metadata
- Support zooming, panning, and graph layout selection (force-directed, hierarchical, radial)
- Support filtering by: `node_type`, `repo`, `community`, `role`, `language`, `edge_type`
- Support search: typing a symbol name highlights matching nodes and dims all others
- Support path highlighting: clicking two nodes draws the shortest path between them
- Show edge direction with arrow heads
- Color-code nodes by type and/or community
- Show edge labels on hover (the edge type)
- Support dark and light themes
- Export selected subgraph as PNG or SVG

### 5.2 — View Types to Generate

**View 1: Full Repository Graph** (`{repo_name}_full.html`)
- All nodes and edges from a single repo
- Clustered by community using force-directed layout with community gravity

**View 2: Multi-Repo Upstream/Downstream Graph** (`multi_repo_flow.html`)
- All repos shown as super-nodes (expandable)
- Cross-repo edges shown prominently in a distinct color (e.g., red for upstream, blue for downstream)
- Clicking a cross-repo edge shows: match method, matched symbols, confidence level
- Layout: hierarchical with upstream repos at top, downstream at bottom

**View 3: End-to-End Flow Graph** (`e2e_flow_{entry_point}.html`)
- Starting from a selected entry point (API endpoint, CLI command), trace ALL reachable nodes
- Highlight the complete execution chain as a colored path
- Show call chain sequentially with step numbers
- Show data flow overlaid as dashed arrows
- Include DB queries, external API calls, and message queue interactions

**View 4: Taint Flow Graph** (`taint_flows.html`)
- Show only taint-relevant nodes and their paths
- Sources in green, sinks in red, sanitizers in yellow, unsanitized paths in bright red
- Each path is a labeled lane

**View 5: Blast Radius View** (generated dynamically via MCP or CLI)
- Centered on a specific node
- Concentric rings showing: direct callees (ring 1), transitive callees (ring 2), cross-repo impact (ring 3)
- Node size proportional to blast radius contribution

**View 6: Dead Code / Technical Debt View** (`dead_code.html`)
- Only `DEAD` role nodes shown
- Grouped by file/module
- Shows last git modification date and author (from git blame)

### 5.3 — HTML Generator Implementation Details

Use `pyvis` for graphs < 5,000 nodes. Switch to embedded `Sigma.js` for larger graphs:

```python
# Pseudocode for html_renderer.py

def generate_graph_html(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    view_type: str,
    output_path: Path,
    config: VisualizationConfig,
) -> None:
    if len(nodes) < 5000:
        _render_pyvis(nodes, edges, view_type, output_path, config)
    else:
        _render_sigma(nodes, edges, view_type, output_path, config)
```

**Node visual properties:**
- Color: mapped from `NodeType` to a fixed color palette (define 30+ distinct colors)
- Shape: `CLASS` → box, `FUNCTION`/`METHOD` → ellipse, `API_ENDPOINT` → diamond, `DB_TABLE` → cylinder (database), `TAINT_SOURCE` → triangle, `TAINT_SINK` → inverted triangle, `COMMUNITY` → star, `REPOSITORY` → hexagon
- Size: proportional to `in_degree + out_degree` (min: 10px, max: 50px)
- Border: dashed for `DEAD` nodes, thick for `CORE` nodes, double for `ENTRY` nodes
- Icon: embed small emoji/unicode in label for quick visual scanning (🔒 for auth, 💾 for DB, 🌐 for API, ⚠️ for taint sink)

**Click interaction (JavaScript panel):**
```javascript
// When node is clicked, show right panel with:
{
  "Node ID": node.node_id,
  "Type": node.node_type,
  "Role": node.role,
  "File": node.file_path + ":" + node.start_line,
  "Signature": node.metadata.signature,
  "Docstring": node.metadata.docstring,
  "Intent": node.metadata.intent_text,
  "Invariants": node.metadata.invariants,
  "Callers (in-edges)": list_with_links,
  "Callees (out-edges)": list_with_links,
  "Blast Radius Count": node.metadata.blast_radius_count,
  "Taint Role": "source/sink/sanitizer/none",
  "Community": node.metadata.community_id,
  "Git Last Modified": node.metadata.git_last_modified,
  "Test Coverage": node.metadata.test_coverage,
}
```

---

## ═══════════════════════════════════════════════════════
## SECTION 6 — MCP SERVER LAYER (AI Agent Interface)
## ═══════════════════════════════════════════════════════

### 6.1 — Server Bootstrap (`mcp_server/server.py`)

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("codegraphengine")
# Register all tools and resources
# Support both stdio (for Claude Code / Cursor) and HTTP SSE (for web agents)
```

**Startup:** `cge mcp --graph ./cge_graph.db --transport stdio`

### 6.2 — Complete MCP Tool Registry

Implement ALL 35 tools below. Each tool must: validate inputs using Pydantic, query the graph engine, and return structured JSON:

#### Graph Navigation Tools
| Tool Name | Description | Key Parameters |
|---|---|---|
| `cge_get_node` | Get full node details by ID or qualified name | `node_id`, `qualified_name` |
| `cge_search_symbols` | Fuzzy search nodes by name | `query`, `node_types[]`, `repo_id`, `limit` |
| `cge_get_neighbors` | Get immediate neighbors of a node | `node_id`, `direction`, `edge_types[]`, `depth` |
| `cge_get_community` | Get community cluster for a node | `node_id` |
| `cge_list_communities` | List all detected communities with summaries | `repo_id` |

#### Call Graph & Flow Tools
| Tool Name | Description | Key Parameters |
|---|---|---|
| `cge_get_call_chain` | Get ordered call chain A→B | `from_node_id`, `to_node_id`, `max_depth` |
| `cge_get_callers` | Get all callers of a function (N levels deep) | `node_id`, `depth` |
| `cge_get_callees` | Get all callees of a function (N levels deep) | `node_id`, `depth` |
| `cge_get_cfg` | Get control flow graph for a function | `function_node_id` |
| `cge_get_entry_to_exit` | Trace full path from API entry to DB/response | `entry_node_id` |
| `cge_get_end_to_end_flow` | Complete E2E flow including cross-repo hops | `entry_node_id` |

#### Data Flow Tools
| Tool Name | Description | Key Parameters |
|---|---|---|
| `cge_get_data_flow` | Trace mutations of a variable | `variable_node_id` |
| `cge_get_taint_paths` | Get all taint paths for source or sink | `node_id`, `direction`, `vuln_class` |
| `cge_get_all_sources` | List all taint sources in graph | `repo_id` |
| `cge_get_all_sinks` | List all taint sinks in graph | `repo_id` |
| `cge_check_sanitization` | Check if a taint path is sanitized | `path_id` |

#### Topology & Impact Tools
| Tool Name | Description | Key Parameters |
|---|---|---|
| `cge_get_blast_radius` | Calculate impact of modifying a node | `node_id`, `max_depth` |
| `cge_get_nodes_by_role` | Get all ENTRY / CORE / DEAD / etc. nodes | `role`, `repo_id` |
| `cge_get_dead_code` | List all unreachable/unused symbols | `repo_id`, `language` |
| `cge_get_entry_points` | List all external entry points | `repo_id` |
| `cge_get_api_inventory` | List all API endpoints with types | `repo_id` |
| `cge_get_db_operations` | List all DB read/write operations | `repo_id`, `table_name` |

#### Cross-Repo Tools
| Tool Name | Description | Key Parameters |
|---|---|---|
| `cge_get_upstream_repos` | List repos this repo depends on | `repo_id` |
| `cge_get_downstream_repos` | List repos that depend on this repo | `repo_id` |
| `cge_get_cross_repo_paths` | Flow paths between two repos | `from_repo_id`, `to_repo_id` |
| `cge_get_cross_repo_edges` | All cross-repo dependency edges | `repo_id`, `direction` |
| `cge_simulate_change_impact` | Cross-repo blast radius simulation | `node_id`, `change_type` |

#### Intent & Business Logic Tools
| Tool Name | Description | Key Parameters |
|---|---|---|
| `cge_get_intent` | Get why a node/file was written | `node_id` |
| `cge_get_invariants` | Get all contracts/invariants for a node | `node_id` |
| `cge_check_drift` | Check if code violates its stated intent | `node_id` |
| `cge_get_business_rules` | List all detected business rules | `repo_id` |

#### Visualization Tools
| Tool Name | Description | Key Parameters |
|---|---|---|
| `cge_generate_graph_html` | Generate clickable HTML for a subgraph | `node_ids[]`, `view_type`, `depth` |
| `cge_generate_flow_diagram` | Generate Mermaid/PlantUML for a flow | `from_node_id`, `to_node_id`, `format` |
| `cge_open_graph_browser` | Open generated HTML in system browser | `html_path` |

#### Meta Tools
| Tool Name | Description | Key Parameters |
|---|---|---|
| `cge_get_graph_stats` | Graph summary: node counts by type, edge counts, repo list | none |
| `cge_refresh_graph` | Re-index modified files (incremental) | `repo_id` |

---

## ═══════════════════════════════════════════════════════
## SECTION 7 — CUCUMBER TEST GENERATION LAYER
## ═══════════════════════════════════════════════════════

### 7.1 — Scenario Extraction Strategy

For every `API_ENDPOINT`, `FUNCTION` with `ENTRY` role, and every `EVENT_CONSUMER`, extract test scenarios using the following strategy:

**Step 1 — Parameter Analysis:**
- Extract all function parameters from the graph node's metadata
- For each parameter, classify its type:
  - `string` → generate: empty string, whitespace only, SQL injection string, XSS string, very long string (> 10,000 chars), Unicode characters, null
  - `integer` → generate: 0, -1, MAX_INT, MIN_INT, null
  - `list/array` → generate: empty list, single element, large list (> 10,000 items), null
  - `object/dict` → generate: empty object, object missing required keys, object with extra unknown keys, null
  - `boolean` → generate: true, false, null (for optional)
  - `file/bytes` → generate: empty file, valid file, malformed file, oversized file, file with malicious content
  - `enum` → generate: each valid value + one invalid value + null

**Step 2 — Flow-Based Scenarios:**
- For each `CFG_BRANCH` in the function's CFG, generate a scenario for each branch (happy path vs. each alternative)
- For each `CFG_EXCEPTION`, generate a scenario that triggers that exception
- For each `CFG_LOOP`, generate: zero iterations, one iteration, many iterations (stress)
- For each `TAINT_SINK` reachable from the function, generate a security test scenario

**Step 3 — Business Rule Scenarios:**
- For each `INVARIANT` linked to the function, generate a scenario that upholds it and one that attempts to violate it
- For each `BUSINESS_RULE`, generate a positive and negative scenario

**Step 4 — Cross-Repo Integration Scenarios:**
- For each `CROSS_REPO_CALLS` edge originating from the function, generate: success response scenario, service unavailable scenario, timeout scenario, malformed response scenario

**Step 5 — Data State Scenarios:**
- For each `QUERIES` or `WRITES_TO_DB` edge, generate: empty dataset scenario, large dataset scenario, concurrent modification scenario

### 7.2 — Gherkin Feature File Template

Each generated `.feature` file follows this structure:

```gherkin
# Generated by CodeGraphEngine v{version}
# Source: {repo_id}/{file_path}:{start_line}
# Node: {qualified_name}
# Role: {role}
# Blast Radius: {blast_radius_count} nodes
# Generated At: {timestamp}
# DO NOT EDIT - Re-generate with: cge generate-tests --node {node_id}

@generated @{node_type_tag} @{role_tag}
Feature: {human_readable_feature_name}
  As a {actor}
  I want to {action}
  So that {business_value}

  Background:
    Given the application is running and accessible
    And the database is seeded with test fixtures
    And all dependent services are mocked or available

  # ── HAPPY PATH SCENARIOS ────────────────────────────────────────────

  @happy-path @{language}
  Scenario: Successful {function_name} with valid input
    Given {setup_preconditions}
    When {actor} calls "{qualified_name}" with valid parameters:
      | Parameter        | Value              | Type    |
      | {param1_name}    | {param1_valid_val} | {type1} |
      | {param2_name}    | {param2_valid_val} | {type2} |
    Then the function should return successfully
    And the response should match the expected schema
    And no errors should be logged
    And the database state should reflect {expected_db_change}

  # ── BOUNDARY VALUE SCENARIOS ─────────────────────────────────────────

  @boundary @{param_name}
  Scenario Outline: {function_name} handles boundary values for {param_name}
    Given {setup_preconditions}
    When {actor} calls "{qualified_name}" with <{param_name}> set to "<value>"
    Then the system should respond with <expected_outcome>
    And the response status should be <expected_status>

    Examples:
      | value                    | expected_outcome          | expected_status |
      | {empty_value}            | validation error          | 400             |
      | {null_value}             | null handling response    | 400             |
      | {min_boundary_value}     | success or boundary error | {min_status}    |
      | {max_boundary_value}     | success or boundary error | {max_status}    |
      | {over_max_value}         | overflow/rejection        | 400             |
      | {unicode_value}          | proper unicode handling   | {unicode_status}|

  # ── ERROR PATH SCENARIOS ─────────────────────────────────────────────

  @error-path @negative
  Scenario: {function_name} handles missing required parameters
    Given {setup_preconditions}
    When {actor} calls "{qualified_name}" without providing required parameter "{required_param}"
    Then a validation error should be raised
    And the error message should clearly identify the missing field
    And no partial side effects should occur

  @error-path @exception
  Scenario: {function_name} handles {exception_type} gracefully
    Given {exception_precondition}
    When {actor} calls "{qualified_name}" with parameters that trigger {exception_type}
    Then the exception should be caught and handled
    And a meaningful error response should be returned
    And the system should remain in a consistent state
    And the error should be logged at the appropriate severity level

  # ── SECURITY SCENARIOS (for taint-reachable functions) ───────────────

  @security @sql-injection
  Scenario: {function_name} is protected against SQL injection
    Given an authenticated {actor}
    When {actor} calls "{qualified_name}" with SQL injection payload:
      """
      {sql_injection_sample}
      """
    Then the request should be rejected or sanitized
    And no SQL should be executed against raw user input
    And the injection attempt should be logged as a security event

  @security @xss
  Scenario: {function_name} is protected against XSS
    Given an authenticated {actor}
    When {actor} calls "{qualified_name}" with XSS payload "<script>alert('xss')</script>"
    Then the output should be properly escaped
    And no script should be executable from the response

  # ── INTEGRATION / CROSS-REPO SCENARIOS ───────────────────────────────

  @integration @cross-repo @{downstream_repo_id}
  Scenario: {function_name} handles downstream service {downstream_service} being unavailable
    Given the downstream service "{downstream_service}" is unreachable
    When {actor} calls "{qualified_name}" with valid parameters
    Then the system should respond with a service unavailable error
    And the fallback behavior should activate: {fallback_description}
    And the circuit breaker should trip after {retry_threshold} failures

  @integration @cross-repo @{downstream_repo_id}
  Scenario: {function_name} handles downstream service returning malformed response
    Given the downstream service "{downstream_service}" returns: {malformed_response_json}
    When {actor} calls "{qualified_name}" with valid parameters
    Then the malformed response should be handled gracefully
    And a parsing error should not propagate to the caller
    And the incident should be logged with full context

  # ── CONCURRENCY SCENARIOS ─────────────────────────────────────────────

  @concurrency @stress
  Scenario: {function_name} handles concurrent requests without race conditions
    Given {concurrent_setup}
    When {concurrent_count} concurrent {actor}s call "{qualified_name}" simultaneously
    Then all requests should complete without data corruption
    And the database should remain in a consistent state
    And no deadlocks should occur

  # ── PERFORMANCE SCENARIOS ─────────────────────────────────────────────

  @performance @sla
  Scenario: {function_name} completes within SLA under normal load
    Given the system is under normal load (< {load_threshold} concurrent users)
    When {actor} calls "{qualified_name}" with {typical_payload}
    Then the response time should be below {sla_ms} milliseconds
    And memory usage should not exceed {memory_threshold_mb} MB

  # ── INVARIANT ENFORCEMENT SCENARIOS ──────────────────────────────────

  @invariant @business-rule
  Scenario: {function_name} enforces invariant: {invariant_text}
    Given a state where violating the invariant would be possible
    When {actor} calls "{qualified_name}" with parameters designed to violate "{invariant_text}"
    Then the invariant should be upheld
    And an appropriate error should be returned
    And the violation attempt should be audited
```

### 7.3 — Step Definition Stub Generator

For each generated `.feature` file, generate a companion step definition file:

**Python (Behave) `steps_{feature_name}.py`:**
```python
# Generated by CodeGraphEngine — implement each @step body
from behave import given, when, then
import pytest

@given(u'the application is running and accessible')
def step_app_running(context):
    raise NotImplementedError("Implement: start the application under test")

@when(u'the actor calls "{qualified_name}" with valid parameters')
def step_call_function(context, qualified_name):
    raise NotImplementedError(f"Implement: invoke {qualified_name} via API/SDK")

@then(u'the function should return successfully')
def step_assert_success(context):
    raise NotImplementedError("Implement: assert the response status is 2xx")
# ... (all steps generated)
```

**Also generate stubs for:**
- Java + Cucumber-Java (`@Given`, `@When`, `@Then` annotations)
- JavaScript + Cucumber-JS (`Given`, `When`, `Then` from `@cucumber/cucumber`)

### 7.4 — Test Generation CLI Commands

```bash
# Generate tests for all entry points in a repo
cge generate-tests --repo ./my-service --output ./tests/features/

# Generate tests for a specific function
cge generate-tests --node "com.example.OrderController.createOrder" --output ./tests/

# Generate tests for all taint sinks (security-focused)
cge generate-tests --filter-role TAINT_SINK --output ./tests/security/

# Generate cross-repo integration tests only
cge generate-tests --cross-repo-only --from-repo service-a --to-repo service-b

# Regenerate tests after code change (incremental)
cge generate-tests --changed-only --since HEAD~1
```

---

## ═══════════════════════════════════════════════════════
## SECTION 8 — CLI INTERFACE (Complete Command Specification)
## ═══════════════════════════════════════════════════════

```
CGE — CodeGraphEngine

Usage: cge [OPTIONS] COMMAND [ARGS]...

Commands:
  init            Initialize a new CGE project in current directory
  index           Parse and index one or more repositories
  status          Show indexing status and graph statistics
  graph           Generate interactive visualization HTML
  query           Query the graph from the command line
  mcp             Start the MCP server for AI agent access
  generate-tests  Generate Cucumber .feature files from graph
  report          Generate architectural reports (Markdown/HTML)
  diff            Compare graph state before/after a code change
  audit           Run security audit (taint analysis report)
  export          Export graph as JSON, GraphML, or Cypher statements

Options for `cge index`:
  --repo PATH_OR_URL        Repository to index (repeatable for multi-repo)
  --repos YAML_FILE         Manifest file for multi-repo setup
  --branch TEXT             Branch to analyze (default: current)
  --commit TEXT             Pin to specific commit
  --languages TEXT          Comma-separated language filter
  --exclude GLOB            Glob patterns to exclude (repeatable)
  --backend [sqlite|neo4j|memory]  Storage backend
  --db-path PATH            Path to SQLite DB (default: ./cge_graph.db)
  --neo4j-uri TEXT          Neo4j connection URI
  --enable-taint            Run taint analysis (slower, more thorough)
  --enable-intent           Extract intent/iCPG tiers
  --enable-llm              Use LLM for semantic enrichment
  --llm-provider [anthropic|openai|ollama]
  --llm-model TEXT
  --community-algo [leiden|louvain|infomap]
  --incremental             Only re-index changed files (uses SHA256 cache)
  --workers INT             Number of parallel parsing workers (default: CPU count)
  --verbose                 Show per-file progress

Options for `cge graph`:
  --view [full|multi-repo|e2e|taint|blast-radius|dead-code|community]
  --node TEXT               Center node for blast-radius/e2e views
  --repos TEXT              Comma-separated repo IDs for multi-repo view
  --output PATH             Output HTML file path
  --max-nodes INT           Max nodes to render (default: 2000; uses sampling above)
  --renderer [pyvis|sigma|d3]  Force specific renderer
  --theme [light|dark]
  --open                    Open in browser after generation

Options for `cge query`:
  --node TEXT               Get node by ID or qualified name
  --callers TEXT            Get callers of a node
  --callees TEXT            Get callees of a node
  --path FROM TO            Get path between two nodes
  --blast-radius TEXT       Get blast radius for a node
  --taint TEXT              Get taint paths for a node
  --dead-code               List all dead code
  --entry-points            List all entry points
  --output [json|table|yaml]
```

---

## ═══════════════════════════════════════════════════════
## SECTION 9 — CONFIGURATION SCHEMA
## ═══════════════════════════════════════════════════════

```yaml
# cge.config.yaml — all values have sensible defaults if omitted

project:
  name: "my-multi-service-platform"
  version: "1.0.0"
  description: "Optional project description for graph metadata"

repositories:
  - path: "./services/order-service"
    role: downstream
    language_hints: [python]
    alias: "order-svc"
  - url: "https://github.com/myorg/auth-service"
    role: upstream
    branch: main
    alias: "auth-svc"
  - path: "./libs/shared-models"
    role: shared-lib
    alias: "shared-models"

storage:
  backend: sqlite                     # sqlite | neo4j | memory
  db_path: "./cge_graph.db"
  neo4j_uri: "bolt://localhost:7687"
  neo4j_user: "neo4j"
  neo4j_password: "${NEO4J_PASSWORD}" # Environment variable substitution

analysis:
  enable_taint: true
  enable_intent: true
  enable_community_detection: true
  community_algorithm: leiden         # leiden | louvain | infomap
  taint_sources:                      # Extend built-in list
    - pattern: "custom_user_input\\("
      language: python
      category: user_input
  taint_sinks:
    - pattern: "legacy_exec\\("
      language: python
      category: command_injection
  node_role_thresholds:
    core_in_degree: 5
    core_out_degree: 5
    utility_in_degree: 2
    utility_out_degree: 1
  max_call_chain_depth: 10
  max_blast_radius_depth: 5
  incremental: true
  workers: 0                          # 0 = use all CPU cores

llm:
  enabled: false                       # Set true to enable semantic enrichment
  provider: anthropic                  # anthropic | openai | ollama
  model: claude-sonnet-4-20250514
  api_key: "${ANTHROPIC_API_KEY}"
  enrich_intents: true
  enrich_communities: true
  max_concurrent_calls: 5

visualization:
  default_renderer: auto              # auto | pyvis | sigma | d3
  sigma_threshold: 5000               # Switch to sigma above this node count
  max_nodes: 2000                     # Sampling limit for large graphs
  default_theme: dark
  color_scheme:
    FUNCTION: "#4A90D9"
    CLASS: "#7B68EE"
    API_ENDPOINT: "#F5A623"
    DB_TABLE: "#7ED321"
    TAINT_SINK: "#D0021B"
    TAINT_SOURCE: "#417505"
    DEAD: "#9B9B9B"
    CORE: "#FF6B35"
    CROSS_REPO_EDGE: "#FF0080"

test_generation:
  output_dir: "./tests/generated"
  languages: [python, java, javascript]
  include_security_scenarios: true
  include_performance_scenarios: true
  include_concurrency_scenarios: false
  sla_ms: 500
  sql_injection_payloads:
    - "' OR '1'='1"
    - "'; DROP TABLE users; --"
    - "1; SELECT * FROM pg_tables"
  xss_payloads:
    - "<script>alert('xss')</script>"
    - "<img src=x onerror=alert(1)>"

mcp:
  transport: stdio                    # stdio | http
  http_port: 8765
  http_host: "127.0.0.1"
  max_result_nodes: 500               # Limit results per tool call to avoid context flooding

parsing:
  exclude_patterns:
    - "**/__pycache__/**"
    - "**/node_modules/**"
    - "**/dist/**"
    - "**/build/**"
    - "**/.git/**"
    - "**/vendor/**"
    - "**/target/**"
    - "**/*.min.js"
    - "**/*.generated.*"
  max_file_size_mb: 5
  follow_symlinks: false
```

---

## ═══════════════════════════════════════════════════════
## SECTION 10 — EDGE CASES, NUKE CASES, AND CORNER HANDLING
## ═══════════════════════════════════════════════════════

Every bullet below is a MANDATORY implementation requirement. Do not treat these as optional.

### 10.1 — Parser Robustness
- **Syntax errors in source files:** Catch all parse exceptions per file. Log warning with file path and error message. Continue indexing remaining files. Never crash the full indexing run on a single bad file.
- **Mixed indentation (Python):** Normalize tabs to 4 spaces before parsing. Log a warning if mixed is detected.
- **Non-UTF-8 encoded files:** Attempt UTF-8, then Latin-1, then CP1252 before giving up. Emit a `ENCODING_ERROR` warning and skip the file.
- **Files > max_file_size_mb:** Skip silently with a count logged at end of run.
- **Empty files (0 bytes):** Create a `FILE` node with metadata `is_empty: true`. Do not attempt parsing.
- **Binary files masquerading as text:** Detect via null byte presence in first 512 bytes. Skip with log.
- **Circular imports:** Detect via cycle detection in the import graph (DFS with visited set). Emit `CIRCULAR_IMPORT` metadata on the cycle nodes. Do NOT crash; continue traversal.
- **Dynamic imports:** `importlib.import_module(var)` where `var` is runtime-determined → create `DYNAMIC_IMPORT` edge with `confidence: HEURISTIC`. Try to resolve via value tracking if the value is a string literal.
- **`__import__` calls, `eval()`-based imports:** Flag as `DYNAMIC_IMPORT` + `TAINT_SINK`.
- **Conditional imports inside `try/except ImportError`:** Parse both branches. Mark as `OPTIONAL_DEPENDENCY`.
- **`TYPE_CHECKING` blocks (Python):** Parse but mark all imports/symbols inside as `type_only: true` — they do not generate runtime `CALLS` edges, only type annotation edges.
- **Overloaded functions (C++/Java/TS):** Create one node per overload signature, with a `OVERLOADS` parent node grouping them.
- **Operator overloading:** Detect `__eq__`, `__add__`, etc. and create `OVERLOADS_OPERATOR` edge.
- **Metaclasses, descriptors, `__getattr__` magic:** Flag the class as `USES_METACLASS` in metadata. Cannot fully trace dynamic attribute resolution; mark affected edges as `confidence: HEURISTIC`.
- **Generated code files** (`.pb.go`, `.generated.ts`, `_pb2.py` protobuf): Detect by filename convention AND by checking for "DO NOT EDIT" comment in first 5 lines. Index them but mark all nodes as `is_generated: true`. Exclude from dead code analysis.
- **Compiled extensions (`.so`, `.pyd`, `.dll`):** Cannot parse. Create `EXTERNAL_DEP` node with name from the import statement. Note: unresolvable.
- **Test files:** Detect by filename patterns (`test_*.py`, `*.spec.ts`, `*Test.java`, files in `tests/` or `__tests__/`). Index them separately, mark all nodes with `is_test: true`. Create `TESTS` edge from test function to the function it tests (by name convention and by `@patch` / `mock` targets).

### 10.2 — Multi-Repo Correlation Edge Cases
- **Same function name in multiple repos:** Disambiguate using `repo_id + qualified_name`. Never conflate nodes from different repos into one node.
- **Monorepo with workspaces:** Treat each workspace as a separate logical repo for graphing purposes but link them explicitly as `CROSS_REPO_CALLS` with `confidence: DETERMINISTIC`.
- **Version conflicts:** If Repo A uses `shared-lib==1.0` and Repo B uses `shared-lib==2.0`, create separate `EXTERNAL_DEP` nodes for each version and note the conflict in metadata.
- **Forked repos:** When two repos have identical commit histories up to a fork point, detect this via `git log` and note the fork point in both repo nodes' metadata. Create a `FORKED_FROM` edge.
- **Private packages:** NPM/PyPI packages that don't resolve to public sources → create `UNRESOLVED_EXTERNAL_DEP` node. Never halt.
- **Circular repo dependencies (A depends on B, B depends on A):** Detect. Emit `CIRCULAR_REPO_DEPENDENCY` warning. Do not crash; complete indexing both, create cross-repo edges, flag the circular cycle in both repo nodes.

### 10.3 — Large Graph Performance
- **> 100,000 nodes:** Switch all graph operations to SQLite-backed (not in-memory NetworkX). Use recursive CTEs for path queries. Warn user that Neo4j backend is recommended.
- **> 1,000,000 nodes:** Require Neo4j or ArangoDB. Refuse to run in-memory mode.
- **Visualization sampling for large graphs:** If `node_count > max_nodes` config threshold, apply a sampling strategy:
  1. Always include all `ENTRY`, `CORE`, `TAINT_SOURCE`, `TAINT_SINK` nodes
  2. Fill remaining slots with highest-degree nodes
  3. Show a visible warning banner in the generated HTML: "Graph sampled: showing {shown}/{total} nodes. Use filters to focus."
- **Incremental indexing:** SHA256-based file change detection is MANDATORY, not optional. Full re-index on an unchanged 10,000-file repo should take < 2 seconds (only file hash checks).
- **Parallel parsing:** Use `concurrent.futures.ProcessPoolExecutor` for CPU-bound AST parsing. Default to `min(32, os.cpu_count() + 4)` workers. Each worker processes one file at a time and writes to a thread-safe queue that the main process batches into the SQLite DB.

### 10.4 — CFG/DFG Edge Cases
- **Unreachable code after `return`:** Detect and create nodes with `is_unreachable: true`. Consider these a variant of dead code.
- **Infinite loops (`while True:`):** Detect. Create `CFG_LOOP` node with `is_infinite: true` metadata. Taint analysis should NOT attempt to trace through an infinite loop. Add a visited-set guard with a max-iteration count of 1.
- **Exception re-raising (`raise` with no argument):** Track re-raise as a pass-through edge from the `except` block to the nearest outer handler or function exit.
- **`finally` blocks:** Always add `CFG_EXIT` path from `finally` block that overrides both normal and exception exits.
- **Generator functions (`yield`):** Create a `GENERATOR_FRAME` virtual node. Model the suspension/resumption as edges between the generator frame and the caller.
- **`async/await` coroutines:** Create `ASYNC_FRAME` nodes. Model `await` as a `SUSPENDS_AT` edge. The event loop is modeled as an `EVENT_LOOP` infrastructure node that all coroutines connect to via `SCHEDULED_BY` edges.
- **Context managers (`with` statements):** Detect `__enter__` and `__exit__` calls. Create `RESOURCE_ACQUIRED` and `RESOURCE_RELEASED` edges flanking the `with` block body.
- **Closures:** A nested function that captures variables from an outer scope creates `CAPTURES` edges from the inner `FUNCTION` node to each captured `VARIABLE` node.

### 10.5 — Taint Analysis Edge Cases
- **Taint through string formatting:** `f"SELECT * FROM {table_name}"` — if `table_name` is tainted, the resulting string IS tainted. Track taint through f-strings, `.format()`, `%` operator, and `+` string concatenation.
- **Taint through collections:** If a tainted value is inserted into a list, dict, or set, all reads from that collection are also tainted. Model via `TAINT_FLOWS` edges through `VARIABLE` nodes.
- **Taint sanitization bypasses:** If a function is named `sanitize_input` but its body is just `return input`, detect this as a **false sanitizer** via intra-function CFG analysis. Do NOT mark it as a sanitizer. Flag it as `SUSPICIOUS_SANITIZER`.
- **Second-order injection:** User input stored to DB (first request) then read and used in SQL (second request) → this is still a taint path, just multi-hop with a `DB_TABLE` node in the middle. Track these as `PERSISTENT_TAINT_PATH`.
- **Deserialization:** `pickle.loads()`, `yaml.load()`, `json.loads()` of tainted data — these ARE taint sinks because deserialization of untrusted data can lead to arbitrary code execution.

### 10.6 — Test Generation Edge Cases
- **Functions with no parameters:** Generate only happy-path and state-dependent scenarios.
- **Functions with > 20 parameters:** Cap generated `Scenario Outline` table at 50 rows. Generate a summary comment: "Partial scenarios generated. {N} parameter combinations omitted."
- **Abstract/interface methods:** Generate tests for the interface contract, not the implementation. Use `@abstract` tag on scenarios.
- **Deprecated functions:** Detect `@deprecated` annotations and `# deprecated` comments. Tag all generated scenarios with `@deprecated`. Add a scenario: "Deprecated function should not be called by new code."
- **Private functions (underscore-prefixed in Python, `private` in Java):** Still generate scenarios but tag with `@internal`. These are unit-testable in isolation.
- **Async functions:** All generated step definitions must use `async def` in Python and `await` in JS. Add a `@async` tag.
- **Functions that return generators/iterators:** Generate scenarios testing first N elements, exhaustion behavior, and exception propagation through `send()`.

---

## ═══════════════════════════════════════════════════════
## SECTION 11 — EXAMPLE USAGE WALKTHROUGH
## ═══════════════════════════════════════════════════════

```bash
# 1. Install
pip install codegraphengine

# 2. Initialize project
cd my-platform
cge init

# 3. Index three related services
cge index \
  --repo ./services/auth-service \
  --repo ./services/order-service \
  --repo ./services/payment-service \
  --repo ./libs/shared-models \
  --enable-taint \
  --enable-intent \
  --workers 8

# 4. View graph statistics
cge status
# Output:
# Nodes: 12,847 | Edges: 58,234
# Repos: 4 (auth-svc, order-svc, payment-svc, shared-models)
# ENTRY nodes: 47 | CORE nodes: 23 | DEAD nodes: 189
# Taint paths: 12 (3 unsanitized, 9 sanitized)
# Communities: 31 | Cross-repo edges: 234

# 5. Generate full multi-repo visualization
cge graph \
  --view multi-repo \
  --output ./graphs/platform_overview.html \
  --theme dark \
  --open

# 6. Trace end-to-end flow from order creation API
cge graph \
  --view e2e \
  --node "order_service.api.orders.create_order" \
  --output ./graphs/order_creation_e2e.html \
  --open

# 7. Generate Cucumber tests for all entry points
cge generate-tests \
  --repo ./services/order-service \
  --output ./tests/features/ \
  --languages python,java

# 8. Start MCP server for AI agents
cge mcp --transport stdio

# 9. Run security audit
cge audit --output ./reports/security_audit.md

# 10. Check blast radius before modifying a function
cge query --blast-radius "shared_models.base.BaseModel.validate" --output json
```

---

## ═══════════════════════════════════════════════════════
## SECTION 12 — IMPLEMENTATION ROADMAP & PHASING
## ═══════════════════════════════════════════════════════

Implement in the following strict order to ensure each phase is independently testable:

### Phase 1 — Core Skeleton (Week 1)
- [ ] Project scaffold, `pyproject.toml`, `click` CLI stub
- [ ] `config.py` with Pydantic schema
- [ ] `repo_loader.py` — local path only
- [ ] `file_walker.py` — Python files only
- [ ] `python_parser.py` — functions and classes only (no CFG yet)
- [ ] `store_sqlite.py` — basic nodes and edges tables
- [ ] `html_renderer.py` — pyvis, no filtering yet
- [ ] **Deliverable:** `cge index ./my_repo && cge graph --output out.html` works end-to-end

### Phase 2 — Analysis Depth (Week 2)
- [ ] `cfg_builder.py` for Python
- [ ] `dfg_builder.py` for Python
- [ ] `call_graph_builder.py` for Python
- [ ] `import_graph_builder.py`
- [ ] `topology_classifier.py`
- [ ] `blast_radius_calculator.py`
- [ ] Extend HTML renderer with click-panel detail view
- [ ] **Deliverable:** Full single-repo Python graph with roles, blast radius, CFG

### Phase 3 — Multi-Language (Week 3)
- [ ] `javascript_parser.py`, `typescript_parser.py`
- [ ] `java_parser.py`
- [ ] `go_parser.py`
- [ ] `generic_treesitter_parser.py` fallback
- [ ] SQL, YAML, proto file parsers
- [ ] **Deliverable:** Polyglot indexing working

### Phase 4 — Security & Taint (Week 4)
- [ ] `taint_analyzer.py`
- [ ] Taint visualization view
- [ ] `cge audit` command
- [ ] Security scenario generation in test generator
- [ ] **Deliverable:** Full taint analysis with security Cucumber scenarios

### Phase 5 — Multi-Repo (Week 5)
- [ ] `multi_repo_correlator.py`
- [ ] `cross_repo_linker.py`
- [ ] `graph_merger.py`
- [ ] Multi-repo visualization view (upstream/downstream)
- [ ] Cross-repo MCP tools
- [ ] **Deliverable:** Multi-repo end-to-end flow graph

### Phase 6 — MCP Server (Week 6)
- [ ] All 35 MCP tools
- [ ] MCP resources
- [ ] `cge mcp` CLI command
- [ ] Test with Claude Code and Cursor
- [ ] **Deliverable:** Fully functional MCP server tested against real AI agents

### Phase 7 — Test Generation (Week 7)
- [ ] `scenario_extractor.py`
- [ ] `input_classifier.py`
- [ ] `gherkin_writer.py`
- [ ] `step_definition_gen.py` for Python, Java, JS
- [ ] `cge generate-tests` CLI command
- [ ] **Deliverable:** Auto-generated Cucumber .feature files for a real codebase

### Phase 8 — Polish & Performance (Week 8)
- [ ] `community_detector.py` with Leiden
- [ ] `intent_extractor.py`
- [ ] Incremental indexing (SHA256 cache)
- [ ] Sigma.js renderer for large graphs
- [ ] Neo4j backend adapter
- [ ] `cge diff` command
- [ ] Comprehensive README and documentation
- [ ] **Deliverable:** Production-ready v1.0.0

---

## ═══════════════════════════════════════════════════════
## SECTION 13 — TESTING CGE ITSELF
## ═══════════════════════════════════════════════════════

CGE must maintain its own test suite covering:

- **Unit tests:** Each parser tested against a fixture file per language. Each analyzer tested with a hand-crafted graph fixture. Each MCP tool tested with a mock query engine.
- **Integration tests:** Index the `examples/sample_repos/` directory. Assert exact node counts, edge counts, role counts.
- **Regression tests:** Known taint paths in sample repos must always be detected. Known dead code nodes must always be classified as `DEAD`.
- **Visualization smoke tests:** All 6 view types must generate valid HTML that passes `html5lib` validation.
- **MCP tool contract tests:** Every tool's JSON schema must validate against the MCP protocol spec.
- **Performance benchmarks:** Index the `cpython` standard library (> 500 Python files). Incremental re-index after touching one file must complete in < 3 seconds.

---

*End of PROMPT — CodeGraphEngine (CGE) Complete Specification*
*Version: 1.0.0 | Compiled from research on: graphify, graphiti, codebadger, mcp-joern, code-intel-mcp, Fraunhofer CPG, TheAuditor, depwire, claude-bootstrap (iCPG), codebase-summary-bot, mco, Optave codegraph, shannon, Joern, ArangoDB, and related ecosystem tools.*
