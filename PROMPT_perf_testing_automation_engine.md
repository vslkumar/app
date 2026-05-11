# MASTER PROMPT: perf-testing-automation-engine (PTAE)
## Codebase Graphification → JMX Generation → BlazeMeter Orchestration

---

> **USAGE INSTRUCTION:** Feed this entire document as the system/user prompt to a capable LLM (Claude Sonnet/Opus, GPT-4o, Gemini Ultra) or use it as the specification document for a development team. Every section maps to a concrete, buildable module. Nothing here is aspirational — everything must be implemented.

---

## ═══════════════════════════════════════════════════════
## SECTION 0 — MISSION STATEMENT AND PRIME DIRECTIVE
## ═══════════════════════════════════════════════════════

You are tasked with architecting and implementing **perf-testing-automation-engine (PTAE)** — a production-grade, open-source Python framework that transforms one or more raw source code repositories into a unified, queryable, multi-dimensional knowledge graph, and then leverages that graph to **automatically generate JMeter (JMX) performance test scripts**, **upload them to BlazeMeter**, **execute load tests**, and **compare current vs previous performance reports** — all orchestrated through a single Model Context Protocol (MCP) server consumable by AI agents.

PTAE has **three primary output contracts** that must all be satisfied simultaneously:

1. **GRAPH CONTRACT:** Correlate and connect multiple code repositories (application code + config code, or direct Bitbucket repos) into a unified upstream/downstream end-to-end flow graph, with every API endpoint, auth boundary, database touchpoint, and cross-repo edge represented as queryable nodes/edges.

2. **JMX CONTRACT:** From the graph, automatically generate fully valid, BlazeMeter-compatible JMeter (`.jmx`) test scripts covering every discovered API endpoint and every end-to-end flow — with realistic parameterization (CSV data sets), authentication handling, response correlation (token extraction), assertions, and configurable load profiles.

3. **BLAZEMETER CONTRACT:** Via a dedicated MCP server, programmatically upload the generated JMX to BlazeMeter, trigger execution, poll for completion, fetch the resulting performance reports, fetch the previous baseline report, and produce a structured **current-vs-previous comparison report** highlighting regressions, improvements, and SLA violations.

**This framework explicitly does NOT include:**
- Cucumber/Gherkin test generation (removed from prior iteration)
- Any functional/unit test generation
- Code modification, refactoring, or patching capabilities

PTAE is laser-focused on **performance testing automation driven by deterministic code-graph analysis**.

---

## ═══════════════════════════════════════════════════════
## SECTION 1 — ARCHITECTURAL OVERVIEW
## ═══════════════════════════════════════════════════════

### 1.1 — System Architecture Diagram (Textual)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                  perf-testing-automation-engine (PTAE)                     │
│                                                                            │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐         │
│  │  INGESTION   │   │   ANALYSIS   │   │      GRAPH STORE       │         │
│  │   LAYER      │──▶│    LAYER     │──▶│  (SQLite / Neo4j /     │         │
│  │              │   │              │   │   NetworkX in-memory)  │         │
│  │ • Code repo  │   │ AST + CFG +  │   │                        │         │
│  │ • Config repo│   │ Call Graph + │   │  Unified schema across │         │
│  │ • Bitbucket  │   │ API Extract +│   │  all repos             │         │
│  │ • Multi-repo │   │ Auth Flow +  │   │                        │         │
│  └──────────────┘   │ Cross-Repo   │   └───────────┬────────────┘         │
│                     │ Linking      │               │                      │
│                     └──────────────┘               │                      │
│                                                    │                      │
│  ┌──────────────────────────────────────────────┐  │                      │
│  │           JMX GENERATION LAYER               │  │                      │
│  │                                              │  │                      │
│  │  • Scenario Extractor (per API/E2E flow)     │◀─┤                      │
│  │  • HTTP Sampler Builder                      │  │                      │
│  │  • Auth Manager Builder (token, OAuth, JWT)  │  │                      │
│  │  • CSV Data Set Generator                    │  │                      │
│  │  • Correlation Extractor (regex / JSONPath)  │  │                      │
│  │  • Assertion Generator                       │  │                      │
│  │  • Thread Group / Load Profile Builder       │  │                      │
│  │  • Timer / Pacing Generator                  │  │                      │
│  └────────────┬─────────────────────────────────┘  │                      │
│               │                                    │                      │
│               ▼                                    │                      │
│  ┌──────────────────────────────────────────────┐  │                      │
│  │       BLAZEMETER INTEGRATION LAYER           │  │                      │
│  │                                              │  │                      │
│  │  • API Client (a.blazemeter.com/api/v4)      │  │                      │
│  │  • Project / Test management                 │  │                      │
│  │  • JMX upload                                │  │                      │
│  │  • Test trigger + polling                    │  │                      │
│  │  • Report fetch + parsing                    │  │                      │
│  │  • Current vs Previous comparison engine     │  │                      │
│  └────────────┬─────────────────────────────────┘  │                      │
│               │                                    │                      │
│  ┌────────────▼────────────────────────────────┐   │   ┌──────────────┐   │
│  │              MCP SERVER                     │◀──┴──▶│ INTERACTIVE  │   │
│  │  (stdio / HTTP SSE)                         │       │  GRAPH UI    │   │
│  │                                             │       │  (React +    │   │
│  │  • Graph query tools                        │       │  Sigma.js +  │   │
│  │  • JMX generation tools                     │       │  Graphology) │   │
│  │  • BlazeMeter orchestration tools           │       │              │   │
│  │  • Comparison report tools                  │       │              │   │
│  └─────────────────────────────────────────────┘       └──────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 — Technology Stack

| Layer | Primary Library | Fallback / Alternative | Purpose |
|---|---|---|---|
| CLI entry | `click` | `typer` | Command-line interface |
| Config / settings | `pydantic-settings` | `dynaconf` | Schema-validated config |
| AST parsing (Python) | `ast` (stdlib) + `tree-sitter-python` | `libcst` | Syntax tree extraction |
| AST parsing (JS/TS) | `tree-sitter-javascript` / `tree-sitter-typescript` | `esprima` via subprocess | JS/TS parsing |
| AST parsing (Java) | `tree-sitter-java` | `javalang` | Java parsing |
| AST parsing (Go) | `tree-sitter-go` | subprocess `go ast` | Go parsing |
| AST parsing (C#) | `tree-sitter-c-sharp` | Roslyn via subprocess | C# parsing |
| AST parsing (Ruby) | `tree-sitter-ruby` | `parser` gem subprocess | Ruby parsing |
| Multi-language universal | `tree-sitter` (Python bindings) | per-language fallbacks | Universal parser |
| Graph engine (local) | `networkx` | `igraph` | In-memory graph computation |
| Graph storage (embedded) | `sqlite3` (stdlib) | `tinydb` | Persistent local storage |
| Graph storage (advanced) | `neo4j` driver | `arangodb` HTTP client | Enterprise graph DB |
| Git integration | `gitpython` | subprocess `git` | Clone, blame, history |
| Bitbucket API | `atlassian-python-api` | direct `httpx` to Bitbucket REST | Bitbucket Cloud/Server access |
| HTTP client | `httpx` (async) | `requests` | API calls (Bitbucket, BlazeMeter) |
| JMX generation | `lxml` (XML build) + custom templates | `xml.etree.ElementTree` | Build valid JMeter XML |
| JMX templating | `jinja2` | direct lxml | Template scaffolding |
| BlazeMeter API | custom client built on `httpx` | — | Test upload, execution, reports |
| Report parsing | `pandas` + `orjson` | `csv` (stdlib) | Parse BlazeMeter CSVs / JSON |
| Report comparison | `pandas` + `scipy.stats` | manual statistics | Statistical regression detection |
| OpenAPI parsing | `prance` + `openapi-spec-validator` | manual YAML parsing | Extract API specs from config repos |
| YAML parsing | `ruamel.yaml` | `pyyaml` | Config files, manifests |
| Concurrency | `concurrent.futures` + `asyncio` | — | Parallel repo ingestion, async API calls |
| Serialization | `orjson` | `ujson` | Fast JSON I/O |
| MCP server | `mcp` (official Python SDK) | `fastapi` SSE fallback | AI agent protocol interface |
| Visualization data layer | `graphology` (via JSON export) | — | React UI data structure |
| Visualization renderer | `sigma.js v3` + `react-sigma` | — | WebGL graph rendering |
| Optional LLM | `anthropic` SDK | `openai` SDK | Semantic enrichment of API names/intents |

### 1.3 — Directory Layout (Generated Project Structure)

```
perf-testing-automation-engine/
├── ptae/
│   ├── __init__.py
│   ├── cli.py                        # Click CLI entry points
│   ├── config.py                     # Pydantic settings & schema
│   │
│   ├── ingestion/                    # Layer 1: Pull & normalize repos
│   │   ├── __init__.py
│   │   ├── repo_loader.py            # Git clone, local path, Bitbucket
│   │   ├── bitbucket_client.py       # Bitbucket Cloud/Server API
│   │   ├── config_repo_resolver.py   # Pair code repo with its config repo
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
│   │   ├── ruby_parser.py
│   │   ├── openapi_parser.py         # OpenAPI/Swagger spec parsing
│   │   ├── postman_parser.py         # Postman collection parsing
│   │   └── generic_treesitter_parser.py
│   │
│   ├── analysis/                     # Layer 3: Graph construction
│   │   ├── __init__.py
│   │   ├── ast_analyzer.py
│   │   ├── api_endpoint_extractor.py # CRITICAL for JMX gen
│   │   ├── auth_flow_extractor.py    # Auth middleware detection
│   │   ├── dto_schema_extractor.py   # Request/response shapes
│   │   ├── db_operation_extractor.py # DB read/write hotspots
│   │   ├── cfg_builder.py
│   │   ├── call_graph_builder.py
│   │   ├── import_graph_builder.py
│   │   ├── cross_repo_linker.py
│   │   ├── topology_classifier.py
│   │   ├── e2e_flow_extractor.py     # End-to-end flow chains
│   │   └── perf_hotspot_detector.py  # Hot paths likely to bottleneck
│   │
│   ├── graph/                        # Layer 4: Graph storage & querying
│   │   ├── __init__.py
│   │   ├── schema.py                 # Node/Edge dataclass definitions
│   │   ├── store_sqlite.py
│   │   ├── store_neo4j.py
│   │   ├── store_networkx.py
│   │   ├── query_engine.py
│   │   ├── path_finder.py
│   │   └── graph_merger.py
│   │
│   ├── jmx_generation/               # Layer 5: JMX test script generation
│   │   ├── __init__.py
│   │   ├── scenario_extractor.py     # Derive scenarios from graph
│   │   ├── jmx_builder.py            # Master JMX file builder
│   │   ├── thread_group_builder.py   # Load profile / concurrency
│   │   ├── http_sampler_builder.py   # HTTP request samplers
│   │   ├── auth_manager_builder.py   # Token / OAuth / JWT / Basic auth
│   │   ├── csv_dataset_builder.py    # Parameterization data files
│   │   ├── correlation_extractor.py  # Token/ID extraction logic
│   │   ├── assertion_builder.py      # Response assertions
│   │   ├── timer_builder.py          # Think times / pacing
│   │   ├── header_manager_builder.py # HTTP headers
│   │   ├── listener_builder.py       # Result listeners
│   │   ├── load_profile_resolver.py  # SLA-derived load profiles
│   │   ├── data_generator.py         # Test data synthesis
│   │   └── templates/
│   │       ├── jmx_skeleton.xml.j2
│   │       ├── thread_group.xml.j2
│   │       ├── http_sampler.xml.j2
│   │       ├── auth_jwt.xml.j2
│   │       ├── auth_oauth2.xml.j2
│   │       ├── auth_basic.xml.j2
│   │       ├── csv_dataset.xml.j2
│   │       ├── response_assertion.xml.j2
│   │       ├── duration_assertion.xml.j2
│   │       ├── json_path_extractor.xml.j2
│   │       ├── regex_extractor.xml.j2
│   │       ├── constant_timer.xml.j2
│   │       ├── uniform_random_timer.xml.j2
│   │       └── header_manager.xml.j2
│   │
│   ├── blazemeter/                   # Layer 6: BlazeMeter orchestration
│   │   ├── __init__.py
│   │   ├── api_client.py             # Low-level BlazeMeter v4 API
│   │   ├── auth.py                   # API key/secret auth
│   │   ├── project_manager.py        # Workspace/project CRUD
│   │   ├── test_manager.py           # Test entity CRUD
│   │   ├── jmx_uploader.py           # Upload JMX + supporting files
│   │   ├── test_runner.py            # Trigger + poll execution
│   │   ├── report_fetcher.py         # Fetch reports/timeseries
│   │   ├── comparison_engine.py      # Current vs previous diff
│   │   ├── sla_validator.py          # Validate SLAs from config
│   │   └── webhook_handler.py        # Optional: receive BM webhooks
│   │
│   ├── mcp_server/                   # Layer 7: MCP tool endpoints
│   │   ├── __init__.py
│   │   ├── server.py                 # MCP server bootstrap
│   │   └── tools/
│   │       ├── graph_tools.py        # Query graph, neighbors, flows
│   │       ├── flow_tools.py         # E2E flow, call chains
│   │       ├── api_inventory_tools.py# API endpoint discovery
│   │       ├── jmx_tools.py          # Generate/preview JMX
│   │       ├── blazemeter_tools.py   # Upload, run, fetch, compare
│   │       └── visualization_tools.py# Open graph UI
│   │
│   ├── visualization/                # Layer 8: Clickable graph UI
│   │   ├── __init__.py
│   │   ├── json_exporter.py          # Export graph JSON for React UI
│   │   └── ui_server.py              # Serve React UI locally
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                 # Structured logging
│       ├── hash_utils.py             # SHA256 incremental diff
│       ├── file_utils.py
│       └── llm_client.py             # Optional LLM enrichment
│
├── ptae-ui/                          # React + Sigma.js frontend
│   ├── package.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/                          # See SECTION 14 for full spec
│
├── tests/
│   └── ...                           # PTAE's own test suite
├── examples/
│   └── sample_repos/
├── pyproject.toml
├── ptae.config.yaml                  # Example config
└── README.md
```

---

## ═══════════════════════════════════════════════════════
## SECTION 2 — INGESTION LAYER
## ═══════════════════════════════════════════════════════

### 2.1 — Repository Loader (`ingestion/repo_loader.py`)

Implement `RepoLoader` supporting these input modes. All modes normalize into a canonical `RepoManifest` dataclass before passing downstream:

**Input Modes:**
- `--code-repo /path/to/local/dir` — local code directory
- `--code-repo https://github.com/org/repo` — HTTPS clone
- `--code-repo git@bitbucket.org:org/repo.git` — SSH clone
- `--config-repo /path/to/config-dir` — local config repo (env files, OpenAPI specs, secrets templates)
- `--config-repo https://...` — remote config repo
- `--bitbucket-workspace WORKSPACE --bitbucket-repo REPO` — direct Bitbucket pull via API
- `--repos repos.yaml` — YAML manifest listing N repos with `role:` field (`code`, `config`, `upstream`, `downstream`, `shared-lib`, `schema`, `infra`)
- `--branch feature/xyz` — specific branch
- `--commit abc123` — pin to specific commit (for reproducibility)

**`RepoManifest` Schema:**
```python
@dataclass
class RepoManifest:
    repo_id: str                  # Stable hash of source+commit
    local_path: Path              # Resolved local disk path
    source_type: str              # local | github | bitbucket | gitlab | gitea
    remote_url: Optional[str]
    branch: str
    commit_sha: str
    role: str                     # code | config | upstream | downstream | shared-lib | schema | infra
    paired_with: Optional[str]    # For "config" role: repo_id of paired code repo
    language_hints: list[str]
    git_log: list[GitCommit]
    created_at: datetime
```

### 2.2 — Bitbucket Client (`ingestion/bitbucket_client.py`)

Direct Bitbucket integration MUST support both Bitbucket Cloud and Bitbucket Server (Data Center):

```python
class BitbucketClient:
    def __init__(self, base_url: str, auth: BitbucketAuth):
        # auth: app_password | oauth_token | http_access_token
        ...

    async def list_workspaces(self) -> list[Workspace]: ...
    async def list_repositories(self, workspace: str) -> list[Repository]: ...
    async def clone_repo(self, workspace: str, repo: str, target_dir: Path,
                         branch: str = None, commit: str = None) -> Path: ...
    async def get_default_branch(self, workspace: str, repo: str) -> str: ...
    async def list_branches(self, workspace: str, repo: str) -> list[str]: ...
    async def get_repo_metadata(self, workspace: str, repo: str) -> dict: ...
    async def list_pull_requests(self, workspace: str, repo: str,
                                 state: str = "OPEN") -> list[PullRequest]: ...
    async def get_file_content(self, workspace: str, repo: str,
                               path: str, ref: str = None) -> bytes: ...
```

**Authentication modes:**
- `app_password` — Bitbucket Cloud app password (username + password)
- `oauth_token` — OAuth 2.0 bearer token
- `http_access_token` — Bitbucket Server HTTP access token
- `ssh_key` — fallback to git+SSH clone if API unavailable

**Endpoints used (Bitbucket Cloud v2.0):**
- `GET /2.0/workspaces` — list workspaces
- `GET /2.0/repositories/{workspace}` — list repos
- `GET /2.0/repositories/{workspace}/{repo_slug}` — repo metadata
- `GET /2.0/repositories/{workspace}/{repo_slug}/src/{ref}/{path}` — file content
- `GET /2.0/repositories/{workspace}/{repo_slug}/refs/branches` — branches
- Clone via `https://x-token-auth:{token}@bitbucket.org/{workspace}/{repo}.git`

### 2.3 — Config Repo Resolver (`ingestion/config_repo_resolver.py`)

The configuration repository pattern is treated as a first-class concept. PTAE understands that performance testing requires both code and its runtime configuration:

**Config repo discoverable contents (parsed as graph data):**
- `*.env`, `*.env.*`, `.env.example` — environment variables → `ENV_VAR` nodes
- `application.yml`, `application.properties` (Spring Boot) → `CONFIG_KEY` nodes
- `appsettings.json` (ASP.NET) → `CONFIG_KEY` nodes
- `config/*.yaml`, `values.yaml` (Helm) → `CONFIG_KEY` nodes
- `openapi.yaml`, `swagger.json` — full API contract → `API_ENDPOINT` nodes with request/response schemas
- `postman_collection.json` — additional API definitions
- `nginx.conf`, `apache2.conf` — routing rules
- `terraform/*.tf`, `*.tfvars` — infra resource definitions
- `docker-compose.yml`, `k8s/*.yaml` — service topology
- `bruno/`, `insomnia/` — alternative API client collections
- `*.proto` — gRPC contracts
- `.gitlab-ci.yml`, `Jenkinsfile`, `.github/workflows/*.yaml` — CI/CD with possible BlazeMeter integration hints
- `sla.yaml` / `perf-sla.yaml` (PTAE convention) — explicit performance SLAs

**Pairing logic (auto-pair config repo with code repo):**
1. Same parent organization/workspace + suffix matches (e.g., `order-service` ↔ `order-service-config`)
2. Repo description metadata explicitly references the partner repo
3. Config repo contains a `code-repo.yaml` file pointing to the code repo URL
4. User-supplied explicit pairing via CLI (`--code-repo X --config-repo Y`)

### 2.4 — File Walker (`ingestion/file_walker.py`)

Recursively discover all parseable files:
- Respect `.gitignore` rules
- Honor `--exclude` glob patterns
- Skip binary files (null-byte detection)
- Resolve symlinks with cycle detection
- Configurable max file size (default skip > 5MB)

**Language detection priority:**
1. File extension mapping
2. Shebang line
3. Content heuristics (first 512 bytes)
4. Tree-sitter `detect_language` fallback

**Special files always parsed:**
- `*.sql` → extract table names, stored procedures → `DB_TABLE` / `DB_PROCEDURE` nodes
- `*.proto` → gRPC services → `RPC_DEFINITION` nodes
- `*.yaml` / `*.yml` → OpenAPI, K8s, GitHub Actions
- `*.json` → `package.json`, `tsconfig.json`, Postman collections, `appsettings.json`
- `*.toml` → `pyproject.toml`, `Cargo.toml`
- `Dockerfile` / `docker-compose.yml` → service topology → `SERVICE` nodes
- `*.tf` → Terraform resources → `INFRA_RESOURCE` nodes
- `*.env*` → environment variables → `ENV_VAR` nodes

### 2.5 — Multi-Repo Correlator (`ingestion/multi_repo_correlator.py`)

When N repos are loaded, auto-detect shared interfaces via:
1. Matching exported symbols across repos (function names, class names, REST route strings)
2. Matching package manifest declarations (one repo depends on another)
3. Matching OpenAPI/Swagger endpoint definitions to HTTP client call sites in other repos
4. Matching gRPC proto definitions to generated stub imports
5. Matching database table names referenced across repos (ORM models vs raw SQL strings)
6. Matching message queue topic names in publishers vs consumers

All cross-repo matches become `CROSS_REPO_EDGE` typed edges in the unified graph with confidence tier (`DETERMINISTIC`, `HEURISTIC`, `INFERRED`).

---

## ═══════════════════════════════════════════════════════
## SECTION 3 — ANALYSIS LAYER (Graph Construction)
## ═══════════════════════════════════════════════════════

### 3.1 — Universal Node & Edge Schema (`graph/schema.py`)

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
    FUNCTION          = "FUNCTION"
    METHOD            = "METHOD"
    CONSTRUCTOR       = "CONSTRUCTOR"
    VARIABLE          = "VARIABLE"
    PARAMETER         = "PARAMETER"
    CONSTANT          = "CONSTANT"
    DECORATOR         = "DECORATOR"

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
    SCHEMA_MODEL      = "SCHEMA_MODEL"      # ORM models
    DTO               = "DTO"               # Request/response DTO

    # API / RPC — CRITICAL for JMX generation
    API_ENDPOINT      = "API_ENDPOINT"      # REST/GraphQL/gRPC endpoint
    API_ROUTE         = "API_ROUTE"
    HTTP_METHOD       = "HTTP_METHOD"
    REQUEST_SCHEMA    = "REQUEST_SCHEMA"
    RESPONSE_SCHEMA   = "RESPONSE_SCHEMA"
    HTTP_HEADER       = "HTTP_HEADER"
    HTTP_CLIENT_CALL  = "HTTP_CLIENT_CALL"
    RPC_DEFINITION    = "RPC_DEFINITION"
    RPC_CALL          = "RPC_CALL"

    # Auth — CRITICAL for JMX generation
    AUTH_MIDDLEWARE   = "AUTH_MIDDLEWARE"
    AUTH_PROVIDER     = "AUTH_PROVIDER"     # JWT/OAuth/Basic/Custom
    AUTH_TOKEN_FIELD  = "AUTH_TOKEN_FIELD"  # Where token lives (header/cookie/body)
    LOGIN_ENDPOINT    = "LOGIN_ENDPOINT"    # Endpoint that returns auth token
    PROTECTED_ROUTE   = "PROTECTED_ROUTE"

    # Performance-relevant nodes
    CACHE_OPERATION   = "CACHE_OPERATION"   # Redis/Memcached calls
    EXTERNAL_CALL     = "EXTERNAL_CALL"     # 3rd-party API calls
    HEAVY_QUERY       = "HEAVY_QUERY"       # Joins, full scans, N+1 risks
    EVENT_EMITTER     = "EVENT_EMITTER"
    EVENT_CONSUMER    = "EVENT_CONSUMER"
    MESSAGE_QUEUE     = "MESSAGE_QUEUE"

    # Infrastructure
    SERVICE           = "SERVICE"           # Docker/K8s service
    INFRA_RESOURCE    = "INFRA_RESOURCE"
    ENV_VAR           = "ENV_VAR"
    CONFIG_KEY        = "CONFIG_KEY"

    # End-to-end flow
    E2E_FLOW          = "E2E_FLOW"          # Logical end-to-end scenario
    FLOW_STEP         = "FLOW_STEP"         # Ordered step within an E2E_FLOW

    # SLA
    SLA               = "SLA"               # Explicit SLA constraint

    # Cross-repo
    EXTERNAL_DEP      = "EXTERNAL_DEP"
    CROSS_REPO_SYMBOL = "CROSS_REPO_SYMBOL"

class EdgeType(str, Enum):
    # AST relationships
    DEFINES           = "DEFINES"
    CONTAINS          = "CONTAINS"
    INHERITS          = "INHERITS"
    IMPLEMENTS        = "IMPLEMENTS"
    ANNOTATED_BY      = "ANNOTATED_BY"

    # Call graph
    CALLS             = "CALLS"
    CALLS_ASYNC       = "CALLS_ASYNC"
    CALLS_CONDITIONAL = "CALLS_CONDITIONAL"
    RETURNS_TO        = "RETURNS_TO"

    # Data flow
    READS             = "READS"
    WRITES            = "WRITES"
    PASSES            = "PASSES"
    MUTATES           = "MUTATES"
    PROPAGATES        = "PROPAGATES"

    # Control flow
    BRANCHES_TO       = "BRANCHES_TO"
    LOOPS_BACK        = "LOOPS_BACK"
    RAISES            = "RAISES"
    CATCHES           = "CATCHES"

    # Module / dependency
    IMPORTS           = "IMPORTS"
    DEPENDS_ON        = "DEPENDS_ON"
    EXPORTS           = "EXPORTS"

    # API / RPC
    SERVES            = "SERVES"            # SERVICE serves API_ENDPOINT
    CALLS_API         = "CALLS_API"         # HTTP_CLIENT_CALL → API_ENDPOINT
    EXPECTS_HEADER    = "EXPECTS_HEADER"
    EXPECTS_SCHEMA    = "EXPECTS_SCHEMA"
    RETURNS_SCHEMA    = "RETURNS_SCHEMA"
    PUBLISHES_TO      = "PUBLISHES_TO"
    CONSUMES_FROM     = "CONSUMES_FROM"

    # Auth
    PROTECTED_BY      = "PROTECTED_BY"      # API_ENDPOINT → AUTH_MIDDLEWARE
    ISSUES_TOKEN      = "ISSUES_TOKEN"      # LOGIN_ENDPOINT → AUTH_TOKEN_FIELD
    REQUIRES_TOKEN    = "REQUIRES_TOKEN"
    VALIDATES_TOKEN   = "VALIDATES_TOKEN"

    # DB
    QUERIES           = "QUERIES"
    WRITES_TO_DB      = "WRITES_TO_DB"
    MAPS_TO           = "MAPS_TO"

    # Flow
    STEP_OF           = "STEP_OF"           # FLOW_STEP → E2E_FLOW
    PRECEDES          = "PRECEDES"          # Sequential ordering
    DEPENDS_ON_STEP   = "DEPENDS_ON_STEP"

    # SLA
    GOVERNED_BY       = "GOVERNED_BY"       # API_ENDPOINT → SLA

    # Cross-repo
    UPSTREAM_OF       = "UPSTREAM_OF"
    DOWNSTREAM_OF     = "DOWNSTREAM_OF"
    CROSS_REPO_CALLS  = "CROSS_REPO_CALLS"
    CROSS_REPO_IMPORT = "CROSS_REPO_IMPORT"

class NodeRole(str, Enum):
    ENTRY     = "ENTRY"
    CORE      = "CORE"
    UTILITY   = "UTILITY"
    ADAPTER   = "ADAPTER"
    DEAD      = "DEAD"
    LEAF      = "LEAF"
    BRIDGE    = "BRIDGE"
    GATEWAY   = "GATEWAY"
    HOT_PATH  = "HOT_PATH"      # Identified as perf-critical

class ConfidenceLevel(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    INFERRED      = "INFERRED"
    HEURISTIC     = "HEURISTIC"

@dataclass
class GraphNode:
    node_id: str
    repo_id: str
    node_type: NodeType
    name: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    role: Optional[NodeRole] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Metadata keys (especially relevant for JMX gen):
    # - http_method, http_path, path_params, query_params
    # - request_content_type, response_content_type
    # - request_schema_ref, response_schema_ref
    # - auth_required (bool), auth_type (jwt|oauth|basic|api_key|custom)
    # - rate_limit, expected_p95_ms, expected_throughput
    # - example_request_body, example_response_body
    # - depends_on_endpoint (for chained flows)

@dataclass
class GraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    confidence: ConfidenceLevel = ConfidenceLevel.DETERMINISTIC
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 3.2 — API Endpoint Extractor (`analysis/api_endpoint_extractor.py`)

**THIS IS THE MOST CRITICAL ANALYZER FOR JMX GENERATION.** It must produce a complete inventory of every HTTP endpoint with full request/response shape information.

**Python framework detection patterns:**
- **Flask:** `@app.route('/path', methods=['GET','POST'])`, `@blueprint.route(...)`
- **FastAPI:** `@app.get('/path')`, `@router.post(...)`, with Pydantic models in signature → extract full schema
- **Django:** `urlpatterns = [path('...', view_func)]`, `path()`, `re_path()`, plus DRF `@api_view`, ViewSets
- **Tornado, aiohttp, Starlette, Bottle:** equivalent patterns

**JavaScript/TypeScript:**
- **Express:** `app.get('/path', handler)`, `router.use(...)`
- **NestJS:** `@Controller('/path')`, `@Get('/sub')`, `@Post(...)`, with DTO classes
- **Koa, Hapi, Fastify:** equivalent
- **Next.js API routes:** files under `pages/api/` or `app/api/`

**Java:**
- **Spring Boot:** `@RestController`, `@RequestMapping`, `@GetMapping`, `@PostMapping`, `@PathVariable`, `@RequestBody`, `@RequestParam`
- **JAX-RS:** `@Path`, `@GET`, `@POST`
- **Quarkus, Micronaut:** equivalent

**Go:**
- **Gin:** `router.GET("/path", handler)`, `router.Group(...)`
- **Echo, Fiber, chi, gorilla/mux:** equivalent
- Stdlib `http.HandleFunc("/path", h)`

**C#:**
- **ASP.NET Core:** `[HttpGet("/path")]`, `[HttpPost(...)]`, `[FromBody]`, `[FromRoute]`, `[FromQuery]`
- Minimal APIs: `app.MapGet("/path", handler)`

**Ruby:**
- **Rails:** `routes.rb` parsing, controller actions
- **Sinatra:** `get '/path' do ... end`

**OpenAPI/Swagger (from config repo) is the PRIMARY SOURCE OF TRUTH when present:**
- Parse `openapi.yaml`, `swagger.json` via `prance`
- Each `paths.{path}.{method}` becomes an `API_ENDPOINT` node
- Full request/response schemas from `components.schemas` become `REQUEST_SCHEMA` / `RESPONSE_SCHEMA` nodes with `EXPECTS_SCHEMA` / `RETURNS_SCHEMA` edges
- Examples from `examples:` field become `example_request_body` metadata — directly usable in JMX

**For each API_ENDPOINT, extract and store as metadata:**
```python
{
  "http_method": "POST",                          # GET/POST/PUT/PATCH/DELETE/OPTIONS/HEAD
  "http_path": "/api/v1/orders/{order_id}",
  "path_params": [{"name": "order_id", "type": "string", "example": "ord_123"}],
  "query_params": [{"name": "include", "type": "string", "required": false}],
  "headers_required": ["Authorization", "Content-Type", "X-Request-ID"],
  "request_content_type": "application/json",
  "response_content_type": "application/json",
  "request_schema": { ...JSON Schema... },
  "response_schema": { ...JSON Schema... },
  "request_examples": [ {...}, {...} ],
  "response_examples": {"200": {...}, "400": {...}, "404": {...}},
  "auth_required": true,
  "auth_type": "jwt_bearer",
  "rate_limit": "100/min",
  "deprecated": false,
  "tags": ["orders", "v1"],
  "expected_p95_ms": 500,            # From SLA if available
  "expected_p99_ms": 1500,
  "controller_function_node_id": "...", # Link to the actual handler code
}
```

### 3.3 — Auth Flow Extractor (`analysis/auth_flow_extractor.py`)

Determine HOW each endpoint authenticates so JMX can be generated correctly.

**Detection logic per auth type:**

**JWT Bearer Token:**
- Detect via decorator names (`@jwt_required`, `@requires_auth`), middleware (`AuthMiddleware`, `JwtFilter`), or header inspection (`request.headers['Authorization']`, `.replace('Bearer ', '')`)
- Identify the **login endpoint** that issues the token (typically POST `/login`, `/auth/token`, `/oauth/token`)
- Extract from login endpoint's response schema the JSON path to the token (e.g., `$.access_token`, `$.data.token`)
- Store as `AUTH_PROVIDER` node with metadata: `{"type": "jwt", "token_path": "$.access_token", "header_name": "Authorization", "header_format": "Bearer {token}", "login_endpoint_id": "..."}`

**OAuth 2.0:**
- Detect `client_credentials`, `authorization_code`, `password`, `refresh_token` grant patterns
- Identify token endpoint, client_id/client_secret env var references
- Store flow metadata

**API Key:**
- Detect via header inspection (`X-API-Key`, `X-Auth-Token`), query param check, or middleware reading specific header
- Identify the env var or config key that stores the expected key

**Basic Auth:**
- Detect `Authorization: Basic` header decoding
- Identify if credentials come from DB lookup or config

**Custom Token Schemes:**
- Pattern match other Authorization formats
- Store the exact header name and format

**Output:** For every `PROTECTED_ROUTE`, create a `PROTECTED_BY` edge to the relevant `AUTH_PROVIDER` node. Every JMX HTTP Sampler will use this to attach the correct auth manager.

### 3.4 — DTO / Schema Extractor (`analysis/dto_schema_extractor.py`)

Convert framework-specific schemas into language-agnostic JSON Schema for JMX request body generation:

**Pydantic models (Python):**
```python
class OrderCreate(BaseModel):
    customer_id: str
    items: list[OrderItem]
    shipping_address: Address
```
→ Extract field names, types, validators (`min_length`, `regex`), and `Config.schema_extra` examples → store as JSON Schema with examples.

**TypeScript interfaces / classes / Zod schemas / class-validator decorators:**
- Parse interface body, infer field types and optional/required
- For `class-validator`: extract `@IsString()`, `@IsEmail()`, `@MinLength(3)`, `@IsOptional()` → constraints

**Java DTOs:**
- Parse class with Jackson annotations (`@JsonProperty`)
- Bean Validation: `@NotNull`, `@Size`, `@Email`, `@Pattern`

**C# DTOs:**
- Parse class properties with `[Required]`, `[StringLength]`, `[EmailAddress]`, `[RegularExpression]`

**For nested schemas, recurse fully** so JMX can generate realistic nested JSON bodies.

### 3.5 — End-to-End Flow Extractor (`analysis/e2e_flow_extractor.py`)

Identify multi-step business flows that span multiple endpoints. These become JMX **Transaction Controllers** with chained samplers.

**Detection strategies:**
1. **Sequential API calls in code:** Code that calls API1 then uses its response in API2 → create `E2E_FLOW` with two `FLOW_STEP`s
2. **Test code as oracle:** Existing integration tests that call multiple endpoints in sequence
3. **OpenAPI tags grouping:** Endpoints sharing a tag are candidates for grouping
4. **Cross-service call chains:** Service A's endpoint calls Service B's endpoint via HTTP_CLIENT_CALL → forms a 2-step flow
5. **Documented flows:** `flows.yaml` in config repo (PTAE convention) explicitly defining flows

**`flows.yaml` schema (config-driven flows):**
```yaml
flows:
  - name: complete_order_purchase
    steps:
      - call: POST /api/v1/auth/login
        capture:
          - { name: auth_token, json_path: $.access_token }
      - call: POST /api/v1/cart/items
        body_template: { product_id: "${product_id}", quantity: 1 }
      - call: POST /api/v1/orders
        body_template: { payment_method: "credit_card" }
        capture:
          - { name: order_id, json_path: $.id }
      - call: GET /api/v1/orders/${order_id}
        assert:
          - { json_path: $.status, equals: "confirmed" }
    sla:
      p95_total_ms: 2000
      error_rate_pct: 0.5
```

### 3.6 — Performance Hotspot Detector (`analysis/perf_hotspot_detector.py`)

Identify endpoints most likely to be performance-critical, so JMX generation can apply heavier load profiles:

**Heuristics:**
- High in-degree from other endpoints (frequently called downstream)
- Touches > 3 DB tables in single request
- Performs JOINs > 2 tables (parse SQL or ORM query)
- Has N+1 query risk (loop with DB call inside)
- Calls external APIs synchronously
- Has no caching detected
- Marked as `@cache` but cache invalidation logic exists
- Endpoint serves binary data (file upload/download)

Tag matching endpoints with `role: HOT_PATH` and `expected_load_multiplier: 3.0` metadata.

### 3.7 — Other Analyzers (carry over from graph framework)

- **AST Analyzer** — extract functions, classes, imports per language
- **CFG Builder** — control flow graphs per function
- **Call Graph Builder** — inter-procedural call chains
- **Import Graph Builder** — module dependency graph
- **DB Operation Extractor** — parse SQL strings + ORM queries → `DB_TABLE` and `QUERIES`/`WRITES_TO_DB` edges
- **Topology Classifier** — assign `NodeRole` (ENTRY/CORE/UTILITY/DEAD/HOT_PATH/etc.) by graph centrality
- **Cross-Repo Linker** — five strategies as in Section 2.5

---

## ═══════════════════════════════════════════════════════
## SECTION 4 — GRAPH STORAGE LAYER
## ═══════════════════════════════════════════════════════

### 4.1 — SQLite Schema (primary backend)

```sql
CREATE TABLE IF NOT EXISTS repositories (
    repo_id TEXT PRIMARY KEY,
    path TEXT, url TEXT, source_type TEXT,
    branch TEXT, commit_sha TEXT,
    role TEXT, paired_with TEXT,
    language_hints TEXT, ingested_at TEXT
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    repo_id TEXT,
    node_type TEXT, name TEXT, qualified_name TEXT,
    file_path TEXT, start_line INT, end_line INT,
    language TEXT, role TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id TEXT PRIMARY KEY,
    source_id TEXT, target_id TEXT,
    edge_type TEXT, confidence TEXT,
    weight REAL, metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS e2e_flows (
    flow_id TEXT PRIMARY KEY,
    name TEXT,
    repo_id TEXT,
    steps_json TEXT,
    sla_json TEXT
);

CREATE TABLE IF NOT EXISTS file_hashes (
    file_path TEXT PRIMARY KEY,
    sha256 TEXT, last_indexed TEXT
);

CREATE TABLE IF NOT EXISTS jmx_generations (
    generation_id TEXT PRIMARY KEY,
    generated_at TEXT,
    jmx_file_path TEXT,
    csv_data_files_json TEXT,
    scenario_count INT,
    config_json TEXT
);

CREATE TABLE IF NOT EXISTS blazemeter_runs (
    run_id TEXT PRIMARY KEY,
    bzm_master_id TEXT,
    bzm_test_id TEXT,
    bzm_project_id TEXT,
    generation_id TEXT,
    started_at TEXT, completed_at TEXT,
    status TEXT,
    report_summary_json TEXT,
    full_report_path TEXT,
    FOREIGN KEY(generation_id) REFERENCES jmx_generations(generation_id)
);

CREATE INDEX idx_nodes_type ON nodes(node_type);
CREATE INDEX idx_nodes_repo ON nodes(repo_id);
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_runs_test ON blazemeter_runs(bzm_test_id);
```

**Incremental indexing:** SHA256-based file change detection is MANDATORY. Only re-analyze modified files and recompute affected edges.

### 4.2 — Query Engine (`graph/query_engine.py`)

Unified interface that works regardless of backend (SQLite, Neo4j, NetworkX):

```python
class GraphQueryEngine:
    def get_node(self, node_id: str) -> GraphNode: ...
    def get_all_api_endpoints(self, repo_id: str = None) -> list[GraphNode]: ...
    def get_protected_endpoints(self, repo_id: str = None) -> list[GraphNode]: ...
    def get_auth_provider_for(self, endpoint_id: str) -> Optional[GraphNode]: ...
    def get_login_endpoint(self, repo_id: str = None) -> Optional[GraphNode]: ...
    def get_request_schema(self, endpoint_id: str) -> Optional[dict]: ...
    def get_response_schema(self, endpoint_id: str) -> Optional[dict]: ...
    def get_e2e_flows(self, repo_id: str = None) -> list[E2EFlow]: ...
    def get_hot_paths(self, repo_id: str = None) -> list[GraphNode]: ...
    def get_db_operations_for(self, endpoint_id: str) -> list[GraphNode]: ...
    def get_downstream_calls(self, endpoint_id: str) -> list[GraphNode]: ...
    def get_cross_repo_paths(self, from_repo: str, to_repo: str) -> list[list]: ...
    def get_upstream_repos(self, repo_id: str) -> list[str]: ...
    def get_downstream_repos(self, repo_id: str) -> list[str]: ...
    def get_sla_for(self, endpoint_id: str) -> Optional[dict]: ...
    def search_nodes(self, query: str, node_types: list = None) -> list[GraphNode]: ...
    def get_end_to_end_flow(self, entry_node_id: str) -> EndToEndFlow: ...
```

---

## ═══════════════════════════════════════════════════════
## SECTION 5 — JMX TEST SCRIPT GENERATION LAYER
## ═══════════════════════════════════════════════════════

This is the heart of PTAE's value proposition: convert the graph into a fully-formed, BlazeMeter-compatible JMeter test plan.

### 5.1 — JMX File Anatomy (what we generate)

Every generated `.jmx` is a valid JMeter 5.6+ XML test plan with the following hierarchy:

```
TestPlan
├── Arguments (Test Plan Variables)
│   ├── BASE_URL              ${__P(BASE_URL, https://staging.example.com)}
│   ├── ENV                   ${__P(ENV, staging)}
│   └── ...
├── User Defined Variables
├── CSV Data Set Config (login_credentials.csv, test_data.csv, ...)
├── HTTP Cookie Manager (clears between iterations)
├── HTTP Cache Manager
├── HTTP Header Manager (global headers: Content-Type, Accept)
│
├── Setup Thread Group (runs once before main load)
│   ├── HTTP Sampler: Login (POST /auth/login)
│   │   └── JSON Extractor → ${AUTH_TOKEN}
│   └── (any pre-test setup samplers)
│
├── Thread Group: smoke_test (1 user, 1 iteration)
│   └── ... (one HTTP sampler per discovered endpoint)
│
├── Thread Group: load_test (configured concurrency + ramp-up + duration)
│   ├── HTTP Authorization Manager (uses ${AUTH_TOKEN})
│   ├── Transaction Controller: Endpoint_POST_create_order
│   │   ├── Uniform Random Timer (think time 1-3s)
│   │   ├── HTTP Sampler: POST /api/v1/orders
│   │   │   ├── HTTP Header Manager (endpoint-specific)
│   │   │   ├── JSON body (parameterized from CSV)
│   │   │   ├── Response Assertion (status 2xx, body matches)
│   │   │   ├── Duration Assertion (< SLA p95)
│   │   │   └── JSON Extractor → capture order_id for next sampler
│   │   └── ...
│   ├── Transaction Controller: E2E_complete_purchase
│   │   └── ... (chained samplers from E2E_FLOW)
│   └── ...
│
├── Teardown Thread Group (cleanup if needed)
│
└── Listeners (View Results Tree, Aggregate Report, Summary Report)
```

### 5.2 — Scenario Extractor (`jmx_generation/scenario_extractor.py`)

For every input mode, extract a `Scenario` object that drives JMX generation:

```python
@dataclass
class Scenario:
    scenario_id: str
    scenario_type: str           # smoke | single_endpoint | e2e_flow | hot_path
    endpoint_node_ids: list[str] # Ordered for E2E flows
    load_profile: LoadProfile
    auth_required: bool
    auth_provider_node_id: Optional[str]
    pre_setup_steps: list[str]
    assertions: list[Assertion]
    data_sources: list[DataSourceRef]  # CSV files needed
    correlations: list[Correlation]    # Token/ID extraction chains
    timers: list[TimerSpec]

@dataclass
class LoadProfile:
    threads: int                 # Virtual users
    ramp_up_seconds: int
    duration_seconds: int
    loop_count: int              # -1 for infinite (use duration)
    target_throughput_rps: Optional[float]
    constant_throughput: bool

@dataclass
class Correlation:
    source_sampler_name: str
    extractor_type: str          # json_path | regex | xpath | boundary
    extractor_expression: str
    variable_name: str
```

**Scenario generation rules:**
1. **Smoke scenario** — 1 user, 1 iteration, hits every endpoint once (validation pass)
2. **Per-endpoint load scenario** — for each `API_ENDPOINT`, generate a focused load test
3. **E2E flow scenario** — for each `E2E_FLOW`, generate a chained transaction controller
4. **Hot-path stress scenario** — for each `HOT_PATH` endpoint, generate 3× higher load profile

### 5.3 — JMX Builder (`jmx_generation/jmx_builder.py`)

Uses `lxml` to construct the XML programmatically (not just string templates, which break easily):

```python
from lxml import etree

class JmxBuilder:
    def __init__(self, config: JmxConfig):
        self.config = config
        self.root = etree.Element(
            "jmeterTestPlan",
            version="1.2",
            properties="5.0",
            jmeter="5.6.3"
        )
        self.test_plan_hash_tree = etree.SubElement(self.root, "hashTree")
        self._init_test_plan()

    def _init_test_plan(self):
        tp = etree.SubElement(self.test_plan_hash_tree, "TestPlan",
                              guiclass="TestPlanGui",
                              testclass="TestPlan",
                              testname=self.config.test_name,
                              enabled="true")
        # ... full TestPlan element setup
        # User Defined Variables, Test Plan Args, etc.

    def add_thread_group(self, scenario: Scenario) -> "ThreadGroupBuilder":
        # Delegate to ThreadGroupBuilder
        return ThreadGroupBuilder(self.test_plan_hash_tree, scenario)

    def add_setup_thread_group(self, login_endpoint: GraphNode) -> "SetupThreadGroupBuilder":
        # Login sampler that runs once and captures auth token
        ...

    def add_csv_dataset(self, csv_path: str, variables: list[str], delimiter: str = ","):
        ...

    def add_http_cookie_manager(self):
        ...

    def add_http_cache_manager(self):
        ...

    def add_listeners(self):
        # Aggregate Report, Summary Report (lightweight, BlazeMeter-compatible)
        ...

    def serialize(self, output_path: Path):
        xml_bytes = etree.tostring(self.root, pretty_print=True,
                                   xml_declaration=True, encoding="UTF-8")
        output_path.write_bytes(xml_bytes)
```

### 5.4 — HTTP Sampler Builder (`jmx_generation/http_sampler_builder.py`)

For each `API_ENDPOINT` node, generate a JMeter `HTTPSamplerProxy`:

**Critical fields populated from graph metadata:**
- `HTTPSampler.domain` → from `BASE_URL` variable
- `HTTPSampler.port` → empty (use URL default)
- `HTTPSampler.protocol` → `https` (default)
- `HTTPSampler.path` → from endpoint `http_path`, with path params templated (`/api/v1/orders/${order_id}`)
- `HTTPSampler.method` → from `http_method`
- `HTTPSampler.contentEncoding` → `UTF-8`
- `HTTPSampler.follow_redirects` → `true`
- `HTTPSampler.auto_redirects` → `false`
- `HTTPSampler.use_keepalive` → `true`
- `HTTPSampler.DO_MULTIPART_POST` → `true` if endpoint accepts `multipart/form-data`
- **Arguments** → parameterized body or form data (see CSV Data Set rules below)
- **Embedded resources** → `false` (we don't crawl)

For request bodies of POST/PUT/PATCH:
- If `request_content_type == "application/json"`:
  - Build a JSON template from `request_schema` and `request_examples`
  - Replace dynamic fields with `${VAR_NAME}` references to CSV columns or extracted correlations
- If `application/x-www-form-urlencoded`: build name=value pairs
- If `multipart/form-data`: build form parts with file references where applicable

### 5.5 — Auth Manager Builder (`jmx_generation/auth_manager_builder.py`)

For every `Scenario` requiring auth, attach the correct JMeter auth construct based on the `AUTH_PROVIDER` node's metadata:

**JWT Bearer Token flow (most common):**
1. In the **Setup Thread Group**, add a single HTTP Sampler that hits the `LOGIN_ENDPOINT`
2. Body: `{"username": "${USERNAME}", "password": "${PASSWORD}"}` (sourced from CSV)
3. Add a **JSON Extractor** post-processor:
   - JSON Path: from `AUTH_PROVIDER.metadata.token_path` (e.g., `$.access_token`)
   - Variable name: `AUTH_TOKEN`
   - Default value: `NOT_FOUND`
4. In the main Thread Group, add an **HTTP Header Manager** with:
   - `Authorization: Bearer ${AUTH_TOKEN}`

**OAuth 2.0 Client Credentials:**
- Setup sampler hits token endpoint with `grant_type=client_credentials&client_id=...&client_secret=...`
- Extract `access_token`, same pattern

**Basic Auth:**
- Add `HTTPAuthManager` with username/password sourced from CSV

**API Key:**
- Add Header Manager with `X-API-Key: ${API_KEY}` (from config repo's `.env`)

**Custom schemes:**
- Inspect `AUTH_PROVIDER.metadata.header_format` and adapt

**Token refresh:** If detected (refresh endpoint exists in graph), add a JSR223 PreProcessor that checks token expiry and refreshes if needed.

### 5.6 — CSV Data Set Generator (`jmx_generation/csv_dataset_builder.py`)

For every parameter that varies per request (user IDs, product IDs, search queries, etc.), generate a CSV file and a corresponding CSV Data Set Config element.

**Data sourcing strategy (in priority order):**
1. **From `request_examples` in OpenAPI/Postman:** real-looking values
2. **From `default` and `enum` constraints in JSON Schema:** known-good values
3. **From config repo `test-data/*.csv`** files: user-curated test data
4. **From `ptae.config.yaml` `test_data_generators` section:** rules like "generate 1000 UUIDs", "generate 500 random emails"
5. **Faker-based synthetic data:** as fallback, use `faker` library to generate realistic values matching field types and constraints

**Generated CSV file structure** (`./jmx_output/data/`):
```
login_credentials.csv          (username,password)
order_payloads.csv             (customer_id,product_id,quantity,shipping_zip)
user_uuids.csv                 (user_id)
search_queries.csv             (query_text)
```

**CSV Data Set Config XML:**
- `filename` → path relative to JMX file (or BlazeMeter-uploaded data file reference)
- `variableNames` → comma-separated CSV header names
- `delimiter` → `,`
- `quotedData` → `true`
- `recycle` → `true`
- `stopThread` → `false`
- `shareMode` → `shareMode.all` (or `shareMode.thread` for per-user isolation)

### 5.7 — Correlation Extractor (`jmx_generation/correlation_extractor.py`)

For E2E flows where step N depends on data from step N-1, generate post-processors:

**Extractor types:**
- **JSON Extractor:** for `application/json` responses → `$.id`, `$.data.token`, etc.
- **Regex Extractor:** for HTML/text responses
- **XPath2 Extractor:** for XML responses
- **Boundary Extractor:** for raw text with stable left/right anchors

**Auto-detection:** when step N's request template references `${some_var}` and step N-1's response schema contains a field that semantically matches (`id`, `order_id`, `session_token`), automatically wire an extractor.

### 5.8 — Assertion Builder (`jmx_generation/assertion_builder.py`)

Every HTTP Sampler gets assertions to validate correctness even under load:

**Response Assertion (default for every sampler):**
- Apply to: `Main sample only`
- Field to test: `Response code`
- Pattern matching: `Equals`
- Patterns to test: `200|201|202|204` (or whatever `2xx` status codes the endpoint can legitimately return per OpenAPI)
- Custom failure message: `"Endpoint {http_method} {http_path} returned non-2xx"`

**Duration Assertion (for SLA enforcement):**
- Duration in milliseconds: from `expected_p95_ms` metadata
- If no SLA defined, default: `5000ms`

**JSON Path Assertion (for response body validation):**
- For every required field in `response_schema`, add a JSON path existence check
- For fields with `enum` constraints, validate value is in enum

**Size Assertion** (optional, for endpoints returning known-size payloads).

### 5.9 — Timer Builder (`jmx_generation/timer_builder.py`)

Realistic load requires realistic pacing:

**Per-sampler think time** (`Uniform Random Timer`):
- Constant delay: 1000ms
- Random delay: 2000ms
- → 1–3 second think time between user actions

**Throughput shaping** (`Constant Throughput Timer`):
- For endpoints with a target RPS in `load_profile.target_throughput_rps`, add a timer
- Calculate target throughput: `target_rps * 60 / threads` samples/min per thread

**Synchronizing Timer** (for burst tests):
- Group `N` threads to fire together
- Use only for explicit "thundering herd" scenarios

### 5.10 — Load Profile Resolver (`jmx_generation/load_profile_resolver.py`)

Source of truth for `threads`, `ramp_up`, `duration`:

**Priority order:**
1. CLI override (`--threads 100 --ramp-up 60 --duration 600`)
2. `ptae.config.yaml` `load_profiles` section
3. `perf-sla.yaml` in config repo
4. Heuristic from graph:
   - If `HOT_PATH` role: `threads = 3 × default`
   - If `LOGIN_ENDPOINT`: `threads = max(10, default × 0.5)` (logins are cheap, but stagger them)
   - Else: `threads = 50`, `ramp_up = 60s`, `duration = 600s`

**Pre-defined profile presets:**
```yaml
load_profiles:
  smoke:        { threads: 1,   ramp_up: 1,    duration: 60,    loop_count: 1 }
  baseline:     { threads: 50,  ramp_up: 60,   duration: 600 }
  load:         { threads: 200, ramp_up: 120,  duration: 1800 }
  stress:       { threads: 500, ramp_up: 300,  duration: 1800 }
  spike:        { threads: 1000, ramp_up: 10,  duration: 300 }
  soak:         { threads: 100, ramp_up: 300,  duration: 14400 }   # 4 hour soak
  endurance:    { threads: 50,  ramp_up: 600,  duration: 43200 }   # 12 hours
```

### 5.11 — JMX Output Structure

The generator produces a self-contained deliverable folder:

```
./jmx_output/
├── test_plan.jmx                  ← Main JMX file
├── data/
│   ├── login_credentials.csv
│   ├── order_payloads.csv
│   └── ...
├── lib/                           ← Optional JMeter plugins/JARs
├── ptae_manifest.json             ← Generation metadata for traceability
└── README_GENERATED.md            ← Auto-generated usage instructions
```

The `ptae_manifest.json` traces every JMX element back to its source graph node:
```json
{
  "ptae_version": "1.0.0",
  "generated_at": "2026-05-11T10:00:00Z",
  "source_graph": { "db_path": "./ptae_graph.db", "graph_hash": "..." },
  "scenarios": [
    {
      "scenario_id": "S001",
      "scenario_type": "single_endpoint",
      "source_endpoint_node_id": "n_abc123",
      "source_endpoint_qualified_name": "order_service.api.orders.create_order",
      "source_file": "src/api/orders.py:42",
      "jmx_thread_group_name": "TG_create_order",
      "load_profile": "baseline",
      "data_sources": ["order_payloads.csv"]
    }
  ]
}
```

This manifest is critical: it makes JMX **explainable** and **regeneratable** when code changes.

---

## ═══════════════════════════════════════════════════════
## SECTION 6 — BLAZEMETER INTEGRATION LAYER
## ═══════════════════════════════════════════════════════

### 6.1 — BlazeMeter API Client (`blazemeter/api_client.py`)

BlazeMeter REST API base: `https://a.blazemeter.com/api/v4`

**Authentication:** HTTP Basic Auth with `API_KEY:API_SECRET` (from BlazeMeter user profile → API Keys)

```python
class BlazeMeterClient:
    def __init__(self, api_key: str, api_secret: str,
                 base_url: str = "https://a.blazemeter.com/api/v4",
                 timeout: float = 30.0):
        self.auth = httpx.BasicAuth(api_key, api_secret)
        self.client = httpx.AsyncClient(
            base_url=base_url, auth=self.auth, timeout=timeout
        )

    # — Workspace / Account / Project —
    async def get_account_info(self) -> dict: ...
    async def list_workspaces(self) -> list[dict]: ...
    async def list_projects(self, workspace_id: int) -> list[dict]: ...
    async def get_or_create_project(self, workspace_id: int, name: str) -> dict: ...

    # — Tests —
    async def list_tests(self, project_id: int) -> list[dict]: ...
    async def get_test(self, test_id: int) -> dict: ...
    async def create_test(self, project_id: int, name: str,
                          configuration: dict) -> dict: ...
    async def update_test(self, test_id: int, configuration: dict) -> dict: ...
    async def delete_test(self, test_id: int) -> None: ...

    # — File upload (JMX, CSV data files, JARs) —
    async def upload_file(self, test_id: int, file_path: Path,
                          file_name: str = None) -> dict: ...
    async def list_test_files(self, test_id: int) -> list[dict]: ...
    async def delete_test_file(self, test_id: int, file_id: int) -> None: ...

    # — Test execution —
    async def start_test(self, test_id: int) -> dict:
        # Returns the master_id for this run
        ...
    async def stop_test(self, master_id: int) -> dict: ...
    async def get_master_status(self, master_id: int) -> dict: ...
    async def get_master(self, master_id: int) -> dict: ...

    # — Reports —
    async def get_summary(self, master_id: int) -> dict: ...
    async def get_aggregate_report(self, master_id: int) -> dict: ...
    async def get_timeseries(self, master_id: int,
                             granularity: int = 5) -> dict: ...
    async def get_errors_report(self, master_id: int) -> dict: ...
    async def get_requests_report(self, master_id: int) -> list[dict]: ...
    async def get_thresholds_report(self, master_id: int) -> dict: ...
    async def get_test_history(self, test_id: int,
                               limit: int = 10) -> list[dict]:
        # Used for previous-run comparison
        ...
```

### 6.2 — JMX Uploader (`blazemeter/jmx_uploader.py`)

```python
class JmxUploader:
    def __init__(self, client: BlazeMeterClient):
        self.client = client

    async def upload_test_bundle(
        self,
        project_id: int,
        test_name: str,
        jmx_output_dir: Path,
    ) -> int:
        """
        Uploads a complete JMX test bundle to BlazeMeter:
          1. Create or update the test entity
          2. Upload test_plan.jmx
          3. Upload all CSV files from data/ as supporting files
          4. Upload any JARs from lib/
          5. Configure test settings (engine count, locations, etc.)
        Returns the test_id.
        """
        # Step 1: Check if test exists by name
        existing = await self._find_test_by_name(project_id, test_name)
        if existing:
            test_id = existing["id"]
        else:
            test = await self.client.create_test(
                project_id=project_id,
                name=test_name,
                configuration=self._build_initial_config(),
            )
            test_id = test["id"]

        # Step 2: Upload JMX
        jmx_path = jmx_output_dir / "test_plan.jmx"
        await self.client.upload_file(test_id, jmx_path)

        # Step 3: Upload CSV data files
        data_dir = jmx_output_dir / "data"
        if data_dir.exists():
            for csv_file in data_dir.glob("*.csv"):
                await self.client.upload_file(test_id, csv_file)

        # Step 4: Upload JARs (optional plugins)
        lib_dir = jmx_output_dir / "lib"
        if lib_dir.exists():
            for jar in lib_dir.glob("*.jar"):
                await self.client.upload_file(test_id, jar)

        # Step 5: Finalize configuration (set main JMX, engine count)
        await self.client.update_test(test_id, {
            "scriptType": "jmeter",
            "filename": "test_plan.jmx",
            "configuration": self._build_full_config(),
        })

        return test_id
```

**Test configuration parameters** (from `ptae.config.yaml`):
- `engines` — number of BlazeMeter engines (load generators), default 1
- `concurrency` — total users per engine
- `iterations` — loop count (overrides JMX)
- `holdFor` — sustained load duration override
- `rampUp` — ramp-up override
- `locations` — geographic locations, e.g., `["us-east-1", "eu-west-1"]`
- `enabled` — whether test is enabled to run

### 6.3 — Test Runner (`blazemeter/test_runner.py`)

```python
class TestRunner:
    def __init__(self, client: BlazeMeterClient,
                 poll_interval_s: int = 30):
        self.client = client
        self.poll_interval = poll_interval_s

    async def execute(self, test_id: int,
                      wait_for_completion: bool = True,
                      timeout_s: int = 7200) -> RunResult:
        """
        Start a BlazeMeter test run, poll until completion (or timeout).
        Returns RunResult with master_id, status, duration, and report links.
        """
        start_response = await self.client.start_test(test_id)
        master_id = start_response["result"]["id"]
        started_at = datetime.utcnow()

        if not wait_for_completion:
            return RunResult(
                master_id=master_id, status="QUEUED",
                started_at=started_at, completed_at=None,
                report_url=f"https://a.blazemeter.com/app/#/masters/{master_id}",
            )

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status_resp = await self.client.get_master_status(master_id)
            status = status_resp["result"]["status"]

            if status in ("ENDED", "COMPLETED", "BUSY_AGENT_ABORTED"):
                completed_at = datetime.utcnow()
                summary = await self.client.get_summary(master_id)
                return RunResult(
                    master_id=master_id, status=status,
                    started_at=started_at, completed_at=completed_at,
                    report_url=f"https://a.blazemeter.com/app/#/masters/{master_id}",
                    summary=summary,
                )

            if status in ("ERROR", "FAILED"):
                raise BlazeMeterRunError(f"Test failed: {status_resp}")

            await asyncio.sleep(self.poll_interval)

        raise BlazeMeterTimeoutError(f"Run did not complete in {timeout_s}s")
```

**BlazeMeter status values to handle:**
- `NOT_RUN` → idle
- `QUEUED` → waiting for engine
- `INITIALIZING` → engines starting
- `BOOTING` → JVM bootup
- `DOWNLOADING` → fetching JMX
- `RUNNING` → test executing
- `ENDED` / `COMPLETED` → finished cleanly
- `ABORTED` → user-cancelled
- `FAILED` / `ERROR` → infrastructure failure

### 6.4 — Report Fetcher (`blazemeter/report_fetcher.py`)

After a run completes, fetch the full set of reports for storage and comparison:

```python
class ReportFetcher:
    async def fetch_full_report(self, master_id: int) -> PerformanceReport:
        summary    = await self.client.get_summary(master_id)
        aggregate  = await self.client.get_aggregate_report(master_id)
        timeseries = await self.client.get_timeseries(master_id, granularity=5)
        errors     = await self.client.get_errors_report(master_id)
        requests   = await self.client.get_requests_report(master_id)
        thresholds = await self.client.get_thresholds_report(master_id)

        return PerformanceReport(
            master_id=master_id,
            summary=summary,
            aggregate=aggregate,
            timeseries=timeseries,
            errors=errors,
            requests=requests,
            thresholds=thresholds,
            fetched_at=datetime.utcnow(),
        )

    async def fetch_previous_run(self, test_id: int,
                                  current_master_id: int = None) -> Optional[PerformanceReport]:
        """Fetch the most recent completed run before the current one."""
        history = await self.client.get_test_history(test_id, limit=20)
        candidates = [
            r for r in history
            if r.get("status") in ("ENDED", "COMPLETED")
            and r["id"] != current_master_id
        ]
        if not candidates:
            return None
        return await self.fetch_full_report(candidates[0]["id"])
```

**`PerformanceReport` fields (per endpoint/label):**
- Sample count
- Average response time
- Median (p50)
- p90, p95, p99
- Min, Max
- Error count, Error rate %
- Throughput (req/s)
- Network throughput (KB/s)
- Standard deviation
- Apdex score (if configured)

### 6.5 — Comparison Engine (`blazemeter/comparison_engine.py`)

The most important deliverable: compare the current run to the previous baseline and produce an actionable diff.

```python
@dataclass
class ComparisonResult:
    test_id: int
    current_run: MasterRunRef
    previous_run: MasterRunRef
    label_diffs: list[LabelDiff]
    overall_verdict: str          # "improved" | "regressed" | "neutral"
    regression_count: int
    improvement_count: int
    sla_violations: list[SlaViolation]
    summary_markdown: str         # Human-readable summary
    summary_json: dict            # Machine-readable

@dataclass
class LabelDiff:
    label: str                    # e.g., "POST /api/v1/orders"
    metric_diffs: dict[str, MetricDiff]   # avg, p95, p99, error_rate, throughput
    verdict: str                  # "improved" | "regressed" | "neutral"
    severity: str                 # "minor" | "moderate" | "major" | "critical"

@dataclass
class MetricDiff:
    metric_name: str
    current_value: float
    previous_value: float
    delta_absolute: float
    delta_percent: float
    is_regression: bool
    statistical_significance: float   # p-value from comparison
```

**Regression detection rules** (configurable in `ptae.config.yaml`):

| Metric | Regression threshold | Severity tiers |
|---|---|---|
| `avg_response_time` | +10% | minor: 10-20%, moderate: 20-50%, major: 50-100%, critical: >100% |
| `p95_response_time` | +15% | minor: 15-30%, moderate: 30-60%, major: 60-120%, critical: >120% |
| `p99_response_time` | +20% | minor: 20-40%, moderate: 40-80%, major: 80-150%, critical: >150% |
| `error_rate` | +0.5 percentage points absolute | minor: 0.5-1pp, moderate: 1-3pp, major: 3-10pp, critical: >10pp |
| `throughput` | -10% (lower is regression) | minor: 10-20%, moderate: 20-40%, major: 40-70%, critical: >70% |

**Statistical significance:** for endpoints with > 100 samples per run, apply a Mann-Whitney U test on the response time distributions to confirm regressions are statistically significant, not random variance.

**SLA Validation:**
```python
@dataclass
class SlaViolation:
    label: str
    metric: str             # e.g., "p95_response_time"
    actual_value: float
    sla_threshold: float
    severity: str
```

SLAs are sourced from:
1. `perf-sla.yaml` in the config repo
2. Endpoint-level `expected_p95_ms` metadata on `API_ENDPOINT` graph nodes
3. `ptae.config.yaml` default SLAs

**Comparison report output formats:**
- **Markdown** (`comparison_report.md`) — human-readable, ready for PR comments / Slack
- **JSON** (`comparison_report.json`) — programmatic consumption
- **HTML** (`comparison_report.html`) — rich visual diff with embedded charts
- **JUnit XML** (`comparison_results.xml`) — CI/CD pipeline integration

**Markdown report template excerpt:**
```markdown
# Performance Comparison Report

**Test:** `order-service-perf-test`
**Current Run:** [Master 12345](https://a.blazemeter.com/app/#/masters/12345) — 2026-05-11 14:30 UTC
**Previous Run:** [Master 12340](https://a.blazemeter.com/app/#/masters/12340) — 2026-05-10 14:30 UTC

## Overall Verdict: 🔴 REGRESSED

- 🔴 Regressions: **3** endpoints
- 🟢 Improvements: **2** endpoints
- ⚪ Unchanged: **15** endpoints
- 🚨 SLA Violations: **1**

## 🚨 Critical Findings

### POST /api/v1/orders — REGRESSED (Major)

| Metric | Previous | Current | Δ | Severity |
|---|---|---|---|---|
| Avg Response Time | 145ms | 287ms | **+97.9%** 🔴 | major |
| p95 Response Time | 320ms | 580ms | **+81.3%** 🔴 | major |
| p99 Response Time | 510ms | 1240ms | **+143.1%** 🔴 | critical |
| Error Rate | 0.1% | 0.3% | +0.2pp | minor |
| Throughput (req/s) | 145 | 98 | **-32.4%** 🔴 | moderate |

**SLA Violation:** p95 (580ms) exceeds defined SLA (500ms)
**Likely cause:** Investigate recent changes in `OrderController.createOrder` — see [graph node](http://localhost:7473/#/node/abc123)

## 🟢 Improvements
... (truncated)
```

### 6.6 — SLA Validator (`blazemeter/sla_validator.py`)

Standalone validation pass after every run. Fails the CI pipeline if any critical SLA is violated:

```python
class SlaValidator:
    def validate(self, report: PerformanceReport, sla_config: SlaConfig
                 ) -> SlaValidationResult:
        violations = []
        for label_data in report.aggregate["result"]:
            label = label_data["labelName"]
            sla = sla_config.for_label(label)
            if sla is None:
                continue
            if sla.p95_ms and label_data["95line"] > sla.p95_ms:
                violations.append(SlaViolation(
                    label=label, metric="p95",
                    actual_value=label_data["95line"],
                    sla_threshold=sla.p95_ms,
                    severity=self._compute_severity(label_data["95line"], sla.p95_ms),
                ))
            # ... (p99, error_rate, throughput)
        return SlaValidationResult(violations=violations,
                                    passed=len(violations) == 0)
```

---

## ═══════════════════════════════════════════════════════
## SECTION 7 — MCP SERVER LAYER (AI Agent Interface)
## ═══════════════════════════════════════════════════════

### 7.1 — Server Bootstrap (`mcp_server/server.py`)

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("perf-testing-automation-engine")
```

Startup: `ptae mcp --graph ./ptae_graph.db --transport stdio`

Supports both `stdio` (Claude Code, Cursor) and HTTP SSE (web agents).

### 7.2 — Complete MCP Tool Registry

#### Graph / API Inventory Tools

| Tool Name | Description | Key Parameters |
|---|---|---|
| `ptae_get_graph_stats` | Repos, node/edge counts by type | none |
| `ptae_list_repos` | All ingested repositories | none |
| `ptae_list_api_endpoints` | All discovered API endpoints | `repo_id`, `protected_only`, `tag` |
| `ptae_get_endpoint_details` | Full endpoint metadata + schemas | `endpoint_node_id` |
| `ptae_get_e2e_flows` | All detected end-to-end flows | `repo_id` |
| `ptae_get_hot_paths` | Performance-critical endpoints | `repo_id` |
| `ptae_get_auth_providers` | All discovered auth mechanisms | `repo_id` |
| `ptae_get_login_endpoint` | The endpoint that issues tokens | `repo_id` |
| `ptae_get_cross_repo_paths` | Upstream/downstream flow chains | `from_repo`, `to_repo` |
| `ptae_get_call_chain` | Inter-procedural call chain | `from_node`, `to_node` |
| `ptae_search_symbols` | Fuzzy search nodes by name | `query`, `node_types`, `limit` |

#### JMX Generation Tools

| Tool Name | Description | Key Parameters |
|---|---|---|
| `ptae_generate_jmx` | Generate complete JMX from graph | `repo_id`, `load_profile`, `output_dir`, `include_endpoints[]`, `exclude_endpoints[]` |
| `ptae_generate_jmx_for_endpoint` | Generate JMX for a single endpoint | `endpoint_node_id`, `load_profile` |
| `ptae_generate_jmx_for_flow` | Generate JMX for an E2E flow | `flow_id`, `load_profile` |
| `ptae_preview_jmx_scenarios` | Preview scenarios before generation | `repo_id` |
| `ptae_list_load_profiles` | Available load profile presets | none |
| `ptae_validate_jmx` | Validate generated JMX syntactically | `jmx_path` |
| `ptae_get_jmx_manifest` | Get traceability manifest for a JMX | `jmx_path` |

#### BlazeMeter Tools

| Tool Name | Description | Key Parameters |
|---|---|---|
| `ptae_bzm_list_workspaces` | List BlazeMeter workspaces | none |
| `ptae_bzm_list_projects` | List projects in a workspace | `workspace_id` |
| `ptae_bzm_list_tests` | List tests in a project | `project_id` |
| `ptae_bzm_upload_jmx` | Upload JMX to BlazeMeter | `project_id`, `test_name`, `jmx_output_dir` |
| `ptae_bzm_run_test` | Start a test run | `test_id`, `wait_for_completion` |
| `ptae_bzm_get_run_status` | Check status of a running test | `master_id` |
| `ptae_bzm_stop_test` | Abort a running test | `master_id` |
| `ptae_bzm_get_report` | Fetch full report for a master | `master_id` |
| `ptae_bzm_get_previous_run` | Get last completed run for a test | `test_id`, `before_master_id` |
| `ptae_bzm_compare_runs` | Compare current vs previous | `current_master_id`, `previous_master_id` |
| `ptae_bzm_compare_latest` | Compare latest 2 runs for a test | `test_id` |
| `ptae_bzm_validate_sla` | Run SLA validation on a report | `master_id`, `sla_config_path` |

#### Orchestration / End-to-End Tools

| Tool Name | Description | Key Parameters |
|---|---|---|
| `ptae_full_pipeline` | Index → Generate JMX → Upload → Run → Compare | `code_repo`, `config_repo`, `bzm_project_id`, `load_profile` |
| `ptae_regenerate_and_compare` | Re-gen JMX from updated graph, run, compare to last | `test_id` |

#### Visualization Tools

| Tool Name | Description | Key Parameters |
|---|---|---|
| `ptae_generate_graph_html` | Generate clickable HTML graph | `view_type`, `repo_id`, `node_id` |
| `ptae_open_graph_ui` | Open the React graph viewer | `port` |

### 7.3 — Tool Implementation Pattern

Every MCP tool follows this contract:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("perf-testing-automation-engine")

@mcp.tool()
async def ptae_generate_jmx(
    repo_id: str,
    load_profile: str = "baseline",
    output_dir: str = "./jmx_output",
    include_endpoints: list[str] = None,
    exclude_endpoints: list[str] = None,
) -> dict:
    """Generate a complete JMX test plan from the repository's graph.

    Returns paths to the JMX file, CSV data files, and manifest.
    """
    engine = get_query_engine()
    endpoints = engine.get_all_api_endpoints(repo_id=repo_id)
    if include_endpoints:
        endpoints = [e for e in endpoints if e.qualified_name in include_endpoints]
    if exclude_endpoints:
        endpoints = [e for e in endpoints if e.qualified_name not in exclude_endpoints]

    builder = JmxBuilder(JmxConfig(test_name=f"ptae_{repo_id}",
                                   load_profile=load_profile))
    scenarios = ScenarioExtractor(engine).extract(endpoints)
    for scenario in scenarios:
        builder.add_scenario(scenario)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    builder.serialize(output_path / "test_plan.jmx")

    manifest = builder.get_manifest()
    (output_path / "ptae_manifest.json").write_bytes(orjson.dumps(manifest))

    return {
        "jmx_path": str(output_path / "test_plan.jmx"),
        "data_files": [str(p) for p in (output_path / "data").glob("*.csv")],
        "manifest_path": str(output_path / "ptae_manifest.json"),
        "scenario_count": len(scenarios),
    }
```

---

## ═══════════════════════════════════════════════════════
## SECTION 8 — VISUALIZATION LAYER (Clickable Graphs)
## ═══════════════════════════════════════════════════════

A high-performance React + Sigma.js + Graphology UI is REQUIRED for visualizing the multi-repo upstream/downstream connectivity graph and showing which endpoints are covered by generated JMX scenarios.

**See SECTION 14 below for the complete UI specification** (technology choices, component architecture, performance requirements, etc.). This UI is mandatory — `pyvis` is banned due to its inability to handle large graphs smoothly.

**PTAE-specific views the UI must support:**
1. **Multi-Repo Topology View** — repos as hexagons, cross-repo edges as animated pink curves
2. **API Inventory View** — only `API_ENDPOINT` nodes, colored by HTTP method, sized by request volume in last run
3. **Endpoint Detail View** — clicking an endpoint shows its full call chain, DB touches, SLA, and the JMX sampler that tests it
4. **E2E Flow View** — sequential step animation through a flow
5. **Performance Heat Map View** — endpoints colored by latest p95 vs SLA (green/yellow/red gradient)
6. **Coverage View** — shows which endpoints have JMX coverage vs which don't

---

## ═══════════════════════════════════════════════════════
## SECTION 9 — CLI INTERFACE
## ═══════════════════════════════════════════════════════

```
PTAE — perf-testing-automation-engine

Usage: ptae [OPTIONS] COMMAND [ARGS]...

Commands:
  init               Initialize a new PTAE project in current directory
  index              Parse and index one or more repositories
  status             Show indexing status and graph statistics
  graph              Generate interactive visualization HTML / open UI
  query              Query the graph from the command line
  mcp                Start the MCP server for AI agent access
  generate-jmx       Generate JMX test scripts from the graph
  validate-jmx       Validate a generated JMX file
  bzm                BlazeMeter operations (subcommands below)
  pipeline           Run the full end-to-end pipeline (index → JMX → run → compare)

Subcommands for `ptae bzm`:
  bzm login          Configure BlazeMeter credentials
  bzm list-projects  List BlazeMeter projects
  bzm upload         Upload JMX bundle to BlazeMeter
  bzm run            Trigger a test run
  bzm status         Get status of a running test
  bzm report         Fetch report for a completed run
  bzm compare        Compare current run with previous baseline
  bzm validate-sla   Validate report against defined SLAs

Options for `ptae index`:
  --code-repo PATH_OR_URL    Code repository (repeatable for multi-repo)
  --config-repo PATH_OR_URL  Paired config repository
  --bitbucket-workspace TEXT Bitbucket workspace name
  --bitbucket-repo TEXT      Bitbucket repo slug
  --repos YAML_FILE          Multi-repo manifest file
  --branch TEXT              Branch to analyze
  --commit TEXT              Pin to specific commit
  --languages TEXT           Comma-separated language filter
  --exclude GLOB             Glob patterns to exclude (repeatable)
  --backend [sqlite|neo4j|memory]
  --db-path PATH             Path to SQLite DB (default: ./ptae_graph.db)
  --neo4j-uri TEXT
  --enable-llm               Use LLM for semantic enrichment
  --llm-provider [anthropic|openai|ollama]
  --incremental              Only re-index changed files
  --workers INT              Number of parallel parsing workers
  --verbose

Options for `ptae generate-jmx`:
  --repo TEXT                Source repo ID (use `ptae status` to list)
  --load-profile TEXT        smoke|baseline|load|stress|spike|soak|endurance|custom
  --output-dir PATH          Output directory (default: ./jmx_output)
  --include-endpoint TEXT    Specific endpoint qualified names (repeatable)
  --exclude-endpoint TEXT    Endpoints to skip (repeatable)
  --include-e2e-flows        Include E2E_FLOW transaction controllers
  --hot-paths-only           Only generate scenarios for HOT_PATH endpoints
  --validate                 Validate JMX after generation

Options for `ptae bzm upload`:
  --jmx-dir PATH             Path to jmx_output directory
  --project-id INT           BlazeMeter project ID
  --test-name TEXT           BlazeMeter test name (created if not exists)
  --engines INT              Number of engines (load generators)
  --concurrency INT          Users per engine
  --locations TEXT           Comma-separated location IDs

Options for `ptae bzm run`:
  --test-id INT              BlazeMeter test ID
  --wait                     Wait for completion (default: false)
  --timeout-seconds INT      Max wait time (default: 7200)

Options for `ptae bzm compare`:
  --test-id INT              BlazeMeter test ID
  --current-master-id INT    Current run master ID (default: latest)
  --previous-master-id INT   Previous run (default: previous completed)
  --output-format [markdown|json|html|junit]
  --output PATH              Output file path
  --fail-on-regression       Exit code 1 if regressions detected
  --sla-config PATH          Path to perf-sla.yaml

Options for `ptae pipeline`:
  --code-repo PATH_OR_URL
  --config-repo PATH_OR_URL
  --bzm-project-id INT
  --bzm-test-name TEXT
  --load-profile TEXT
  --wait                     Wait for BlazeMeter run completion
  --auto-compare             Auto-compare with previous run after completion
  --fail-on-regression
```

---

## ═══════════════════════════════════════════════════════
## SECTION 10 — CONFIGURATION SCHEMA
## ═══════════════════════════════════════════════════════

```yaml
# ptae.config.yaml — all values have sensible defaults if omitted

project:
  name: "order-service-perf-suite"
  version: "1.0.0"

repositories:
  - path: "./services/order-service"
    role: code
    alias: "order-svc"
    paired_config: "order-svc-config"
  - path: "./services/order-service-config"
    role: config
    alias: "order-svc-config"
  - url: "https://bitbucket.org/myorg/payment-service"
    role: downstream
    branch: main
    alias: "payment-svc"
  - bitbucket:
      workspace: myorg
      repo_slug: notification-service
      branch: main
    role: downstream
    alias: "notif-svc"

bitbucket:
  base_url: "https://api.bitbucket.org/2.0"   # or self-hosted server URL
  auth_method: app_password                    # app_password | oauth | http_access_token
  username: "${BITBUCKET_USERNAME}"
  password: "${BITBUCKET_APP_PASSWORD}"

storage:
  backend: sqlite
  db_path: "./ptae_graph.db"

analysis:
  enable_llm: false
  community_algorithm: leiden
  incremental: true
  workers: 0   # 0 = all CPU cores

jmx_generation:
  output_dir: "./jmx_output"
  default_load_profile: baseline
  base_url_variable: BASE_URL
  default_base_url: "https://staging.example.com"
  include_smoke_thread_group: true
  include_e2e_flows: true
  enable_response_assertions: true
  enable_duration_assertions: true
  default_think_time:
    constant_ms: 1000
    random_ms: 2000
  test_data_generators:
    user_uuids:
      type: uuid
      count: 10000
      output: data/user_uuids.csv
      column_name: user_id
    fake_emails:
      type: faker
      faker_method: email
      count: 5000
      output: data/emails.csv
      column_name: email

load_profiles:
  smoke:        { threads: 1,    ramp_up: 1,    duration: 60,    loop_count: 1 }
  baseline:     { threads: 50,   ramp_up: 60,   duration: 600 }
  load:         { threads: 200,  ramp_up: 120,  duration: 1800 }
  stress:       { threads: 500,  ramp_up: 300,  duration: 1800 }
  spike:        { threads: 1000, ramp_up: 10,   duration: 300 }
  soak:         { threads: 100,  ramp_up: 300,  duration: 14400 }
  endurance:    { threads: 50,   ramp_up: 600,  duration: 43200 }

blazemeter:
  api_key: "${BLAZEMETER_API_KEY}"
  api_secret: "${BLAZEMETER_API_SECRET}"
  base_url: "https://a.blazemeter.com/api/v4"
  default_workspace_id: 12345
  default_project_id: 67890
  default_engines: 1
  default_locations:
    - us-east-1
  poll_interval_seconds: 30
  run_timeout_seconds: 7200

comparison:
  thresholds:
    avg_response_time_pct: 10
    p95_response_time_pct: 15
    p99_response_time_pct: 20
    error_rate_pp: 0.5            # percentage points absolute
    throughput_pct: -10            # negative = lower is regression
  severity_tiers:
    avg_response_time_pct:
      minor: [10, 20]
      moderate: [20, 50]
      major: [50, 100]
      critical: [100, .inf]
  require_statistical_significance: true
  min_samples_for_significance: 100
  significance_p_value: 0.05

sla:
  defaults:
    p95_ms: 1000
    p99_ms: 3000
    error_rate_pct: 1.0
    min_throughput_rps: 10
  per_endpoint:
    "POST /api/v1/orders":
      p95_ms: 500
      p99_ms: 1500
      error_rate_pct: 0.1
    "GET /api/v1/health":
      p95_ms: 50
      error_rate_pct: 0.0
  fail_on_severity: major          # minor | moderate | major | critical

mcp:
  transport: stdio
  http_port: 8765
  max_result_nodes: 500

parsing:
  exclude_patterns:
    - "**/__pycache__/**"
    - "**/node_modules/**"
    - "**/dist/**"
    - "**/build/**"
    - "**/.git/**"
    - "**/vendor/**"
    - "**/target/**"
  max_file_size_mb: 5
  follow_symlinks: false
```

**`perf-sla.yaml`** (separate file, conventionally in the config repo):
```yaml
slas:
  - endpoint: "POST /api/v1/orders"
    p50_ms: 200
    p95_ms: 500
    p99_ms: 1500
    error_rate_pct: 0.1
    min_throughput_rps: 100
  - endpoint: "GET /api/v1/orders/{order_id}"
    p95_ms: 100
    error_rate_pct: 0.0
```

---

## ═══════════════════════════════════════════════════════
## SECTION 11 — EDGE CASES, NUKE CASES, CORNER HANDLING
## ═══════════════════════════════════════════════════════

Every bullet below is a MANDATORY implementation requirement.

### 11.1 — Parser & Ingestion Robustness
- **Syntax errors in source files:** Catch per file. Log warning. Continue. Never crash full indexing on one bad file.
- **Non-UTF-8 encoded files:** Try UTF-8 → Latin-1 → CP1252. If all fail, log and skip.
- **Files > max size:** Skip silently with end-of-run count.
- **Empty files:** Create `FILE` node with `is_empty: true`. Skip parsing.
- **Binary files masquerading as text:** Skip via null-byte detection.
- **Circular imports:** Detect cycles in import graph. Mark cycle members. Continue.
- **Dynamic imports** (`importlib.import_module(var)`): Mark as `DYNAMIC_IMPORT` with `confidence: HEURISTIC`.
- **Generated code files** (`.pb.go`, `_pb2.py`, `*.generated.ts`): Detect by filename and "DO NOT EDIT" comment. Mark `is_generated: true`. Exclude from JMX generation scenarios (don't generate tests for generated handlers).
- **Compiled extensions** (`.so`, `.pyd`, `.dll`): Cannot parse. Create `EXTERNAL_DEP` node. Note unresolvable.

### 11.2 — Multi-Repo / Bitbucket Edge Cases
- **Bitbucket rate limiting (1000 req/hour):** Implement exponential backoff with jitter. Cache repo metadata locally.
- **Bitbucket Server vs Cloud API differences:** Auto-detect from base URL. Use the correct endpoint shapes.
- **Bitbucket repo requires 2FA / app password:** Clear error message guiding user to create app password with right scopes (`Repositories: Read`).
- **Self-hosted Bitbucket with custom CA cert:** Support `--ca-bundle PATH` option.
- **Repo too large to clone:** Support `--shallow-clone` (`git clone --depth 1`).
- **Config repo missing or empty:** Continue indexing code repo alone, log warning, fall back to code-based API extraction only.
- **Same endpoint defined in both code AND OpenAPI spec:** OpenAPI is source of truth for schemas; code is source of truth for handler location. Merge into single node.
- **OpenAPI spec out of sync with code:** Detect mismatches (endpoint in spec but no handler in code, or vice versa). Emit warning. Don't fail.
- **Circular repo dependencies:** Detect, note in metadata, continue.

### 11.3 — API Endpoint Detection Edge Cases
- **Wildcard routes** (`/api/*`, `/api/{path:.*}`): Create `API_ENDPOINT` node with `is_catchall: true`. JMX generates a sampler with a literal placeholder path.
- **Method-not-explicit handlers** (`@app.route('/path')` with no `methods=` arg): Default to all methods; create one endpoint node per method.
- **Versioned APIs** (`/v1`, `/v2`, `/api/v3`): Treat each version as a separate endpoint.
- **GraphQL endpoints:** Single endpoint typically (`POST /graphql`); inspect resolvers as logical sub-endpoints. JMX generates one sampler per detected query/mutation.
- **WebSocket endpoints:** Detect (`@websocket`, `ws://`, `WebSocketEndpoint`). Note: JMX has limited WebSocket support; emit warning and use JMeter WebSocket Samplers plugin reference in generated lib/ folder.
- **gRPC endpoints:** Mark as `RPC_DEFINITION`. JMX has no native gRPC support; emit warning, optionally include JMeter gRPC plugin reference, or skip (configurable).
- **Server-sent events (SSE):** Detect `text/event-stream` content type. Generate JMX sampler with long timeout and SSE-aware response handling.
- **File upload endpoints:** Generate multipart sampler with reference to a test file fixture in `data/test_files/`.
- **Streaming responses (chunked):** Mark, generate sampler with appropriate timeout.

### 11.4 — JMX Generation Edge Cases
- **Endpoint with no example data and complex nested schema:** Generate a body using JSON Schema defaults + Faker for unconstrained string/integer fields. Log a warning that synthetic data is being used.
- **Endpoint requiring auth but no login endpoint found in graph:** Emit clear error; require user to specify auth manually via `--auth-config auth.yaml`.
- **Endpoint with circular schema reference** (e.g., `Person` has `manager: Person`): Limit recursion depth to 3 when generating sample data.
- **Endpoint where path params appear nowhere in the body / query:** Generate a CSV column for the path param.
- **Endpoints with extremely long URLs from path params:** JMeter has no inherent limit, but warn if > 2000 chars.
- **Endpoints requiring CSRF tokens:** Detect CSRF middleware. Add pre-step to fetch CSRF token from a known endpoint (configurable).
- **Endpoints with rate limiting** (detected via `@rate_limit` or framework patterns): Insert `Constant Throughput Timer` capped at the documented rate.
- **Endpoints that mutate global state** (POST/PUT/DELETE on shared resources): Add a teardown step to clean up created entities, or mark with `non_idempotent: true` metadata and warn user.
- **OpenAPI spec without examples:** Use `default`, then `enum[0]`, then Faker, then literal "string"/`0`/`true`.

### 11.5 — BlazeMeter Integration Edge Cases
- **BlazeMeter API key rotation:** Detect 401 responses, surface clear error directing user to refresh credentials.
- **Free tier limits hit** (e.g., concurrent test limit): Catch `403` with quota message; suggest waiting or upgrading.
- **Test stuck in `INITIALIZING`:** After 10 minutes, fetch detailed logs via API and surface to user. Offer to abort.
- **Engine boot failure:** Status returns `BUSY_AGENT_ABORTED` or similar. Capture full error log, surface, mark run as failed.
- **JMX upload failures** (file too large > 500MB, unsupported format): Validate locally before upload. If too large, suggest splitting test plan.
- **CSV files referenced in JMX but not uploaded:** Pre-flight check before triggering run. Upload all referenced data files.
- **Test report not yet aggregated** (calls within ~30s of run end may return incomplete data): Wait and retry; mark as `pending_aggregation`.
- **Previous run is from a different test version** (JMX was significantly changed): Comparison engine detects label set mismatch; flags new endpoints (no previous baseline) and removed endpoints (no current data) separately rather than failing comparison.
- **Geographic location not available:** Catch error; fall back to default location with warning.
- **Network connectivity loss mid-poll:** Retry with backoff. Don't lose state — persist `master_id` to local SQLite so we can resume.

### 11.6 — Performance Comparison Edge Cases
- **First-ever run** (no previous): Comparison engine outputs "no baseline available" with current metrics as the new baseline.
- **Previous run had errors / partial data:** Comparison engine ignores metrics from labels with `> 50%` error rate in either run.
- **Sample size too small for stats test:** Fall back to simple percentage threshold comparison; note in report.
- **All metrics within thresholds but trend is concerning:** Track historical trend across last 10 runs; flag "drift" if avg p95 has crept up 5% per run over 5+ runs even if no single comparison fails.
- **New endpoint in current run, absent in previous:** Mark as "new endpoint, no baseline" — neither pass nor fail.
- **Endpoint removed from current run:** Mark as "removed" — not a regression.
- **Labels with non-ASCII characters or special chars:** Normalize for comparison; preserve original for display.

### 11.7 — Concurrency & State
- **Multiple `ptae index` runs in parallel on same DB:** Use SQLite WAL mode; file locking with 5s retry. Refuse if another process holds an exclusive lock.
- **MCP server and CLI running simultaneously:** Both read-only by default; mutations go through a single writer process.
- **Long-running BlazeMeter run** (e.g., 12-hour soak): MCP tool can return immediately with `wait_for_completion=False` and a `master_id` for later polling.

---

## ═══════════════════════════════════════════════════════
## SECTION 12 — EXAMPLE USAGE WALKTHROUGH
## ═══════════════════════════════════════════════════════

```bash
# 1. Install
pip install perf-testing-automation-engine

# 2. Initialize project
mkdir my-perf-suite && cd my-perf-suite
ptae init

# 3. Configure BlazeMeter credentials
ptae bzm login
# Prompts for API key + secret, stores in ~/.ptae/credentials

# 4. Index a code repo + its paired config repo
ptae index \
  --code-repo ./services/order-service \
  --config-repo ./services/order-service-config \
  --workers 8 \
  --verbose

# 5. View graph statistics
ptae status
# Output:
# Repos: 2 (order-svc, order-svc-config)
# Nodes: 8,234 | Edges: 41,558
# API Endpoints: 47 (32 protected, 1 login endpoint detected: POST /auth/login)
# E2E Flows: 6
# Hot Paths: 4
# Cross-Repo Edges: 0 (single service indexed)

# 6. Generate JMX with baseline load profile
ptae generate-jmx \
  --repo order-svc \
  --load-profile baseline \
  --output-dir ./jmx_output \
  --include-e2e-flows \
  --validate

# Output:
# Generated: ./jmx_output/test_plan.jmx
# Data files: 8 CSV files in ./jmx_output/data/
# Scenarios: 53 (47 endpoint + 6 e2e flow)
# Validation: PASSED

# 7. Upload to BlazeMeter
ptae bzm upload \
  --jmx-dir ./jmx_output \
  --project-id 67890 \
  --test-name "order-service-baseline" \
  --engines 2 \
  --locations us-east-1,eu-west-1
# Output:
# Test created: ID 123456
# JMX uploaded: test_plan.jmx (87 KB)
# Data files uploaded: 8
# View test: https://a.blazemeter.com/app/#/projects/67890/tests/123456

# 8. Run the test and wait for completion
ptae bzm run --test-id 123456 --wait --timeout-seconds 1200
# Output:
# Run started: Master ID 9876543
# Status: INITIALIZING → BOOTING → DOWNLOADING → RUNNING (10:00)
# ...
# Status: ENDED (after 10:32)
# Report: https://a.blazemeter.com/app/#/masters/9876543

# 9. Compare current run with previous baseline
ptae bzm compare \
  --test-id 123456 \
  --output-format markdown \
  --output ./reports/comparison_$(date +%Y%m%d).md \
  --fail-on-regression \
  --sla-config ./perf-sla.yaml

# Output:
# Comparing Master 9876543 (current) vs Master 9876510 (previous)
# Overall: 🔴 REGRESSED
# Regressions: 3 | Improvements: 2 | SLA Violations: 1
# Report written: ./reports/comparison_20260511.md
# Exit code: 1 (regressions detected)

# 10. Open the interactive graph UI
ptae graph --view multi-repo --open
# Opens http://localhost:7473 with the React+Sigma UI

# 11. Start the MCP server for AI agent integration
ptae mcp --transport stdio
# Now usable from Claude Code, Cursor, etc.

# 12. End-to-end pipeline in one command
ptae pipeline \
  --code-repo ./services/order-service \
  --config-repo ./services/order-service-config \
  --bzm-project-id 67890 \
  --bzm-test-name "order-service-baseline" \
  --load-profile baseline \
  --wait \
  --auto-compare \
  --fail-on-regression
```

---

## ═══════════════════════════════════════════════════════
## SECTION 13 — IMPLEMENTATION ROADMAP
## ═══════════════════════════════════════════════════════

### Phase 1 — Core Skeleton (Week 1)
- [ ] Project scaffold, `pyproject.toml`, `click` CLI stub
- [ ] `config.py` with Pydantic schema
- [ ] `repo_loader.py` — local path + GitHub HTTPS clone
- [ ] `file_walker.py` — Python files only
- [ ] `python_parser.py` — functions + classes
- [ ] `store_sqlite.py` — basic nodes/edges tables
- [ ] **Deliverable:** `ptae index ./my_repo && ptae status` works

### Phase 2 — API Endpoint Extraction (Week 2)
- [ ] `api_endpoint_extractor.py` — Flask, FastAPI, Express, Spring Boot
- [ ] `dto_schema_extractor.py` — Pydantic, Java DTOs, TypeScript interfaces
- [ ] `auth_flow_extractor.py` — JWT, Basic, API Key, OAuth detection
- [ ] `openapi_parser.py` — OpenAPI/Swagger ingestion
- [ ] **Deliverable:** `ptae status` lists all API endpoints with schemas

### Phase 3 — Bitbucket & Multi-Repo (Week 3)
- [ ] `bitbucket_client.py` — Cloud + Server support
- [ ] `config_repo_resolver.py` — pairing logic
- [ ] `multi_repo_correlator.py`
- [ ] `cross_repo_linker.py`
- [ ] `graph_merger.py`
- [ ] **Deliverable:** Multi-repo + Bitbucket ingestion working

### Phase 4 — JMX Generation Core (Week 4-5)
- [ ] JMX skeleton templates in `templates/`
- [ ] `jmx_builder.py` master assembler
- [ ] `http_sampler_builder.py`
- [ ] `auth_manager_builder.py`
- [ ] `csv_dataset_builder.py` + `data_generator.py`
- [ ] `assertion_builder.py`
- [ ] `timer_builder.py`
- [ ] `thread_group_builder.py`
- [ ] `scenario_extractor.py`
- [ ] `ptae generate-jmx` CLI
- [ ] JMX validation (open with `jmeter --validate` if available, else lxml schema validation)
- [ ] **Deliverable:** Valid JMX files generated for any indexed repo

### Phase 5 — E2E Flow & Correlation (Week 6)
- [ ] `e2e_flow_extractor.py`
- [ ] `correlation_extractor.py`
- [ ] Transaction Controller generation
- [ ] `flows.yaml` config support
- [ ] **Deliverable:** Multi-step flows generate as chained samplers

### Phase 6 — BlazeMeter Integration (Week 7)
- [ ] `api_client.py` — full v4 API coverage
- [ ] `jmx_uploader.py`
- [ ] `test_runner.py` with polling
- [ ] `report_fetcher.py`
- [ ] All `ptae bzm` CLI commands
- [ ] **Deliverable:** Full upload → run → fetch cycle working

### Phase 7 — Comparison & SLA (Week 8)
- [ ] `comparison_engine.py` with statistical tests
- [ ] `sla_validator.py`
- [ ] All report output formats (md, json, html, junit)
- [ ] `ptae bzm compare` CLI
- [ ] **Deliverable:** Production-ready comparison reports

### Phase 8 — MCP Server (Week 9)
- [ ] All MCP tools registered
- [ ] `ptae mcp` CLI
- [ ] Test against Claude Code, Cursor
- [ ] **Deliverable:** AI agents can orchestrate full pipeline via MCP

### Phase 9 — Visualization UI (Week 10)
- [ ] React + Sigma.js + Graphology app (see SECTION 14)
- [ ] Multi-repo topology, API inventory, E2E flow, heat map, coverage views
- [ ] `ptae graph` CLI integration
- [ ] **Deliverable:** Clickable, performant graph UI

### Phase 10 — Polish & CI/CD (Week 11-12)
- [ ] More language parsers (Go, C#, Ruby)
- [ ] GitHub Actions + Jenkins integration examples
- [ ] Comprehensive README and documentation
- [ ] Tutorial videos / example projects
- [ ] **Deliverable:** v1.0.0 release

---

## ═══════════════════════════════════════════════════════
## SECTION 14 — HIGH-PERFORMANCE VISUALIZATION ENGINE
## (REPLACES PYVIS — MANDATORY FOR ALL GRAPH RENDERING)
## ═══════════════════════════════════════════════════════

> **WHY THIS SECTION EXISTS:**
> `Pyvis` is a thin DOM-based wrapper over `vis-network` with no WebGL support, no worker-threaded layouts, and catastrophic performance above ~500 nodes. It is **BANNED** as a production renderer for PTAE. This section specifies a modern, WebGL-first, React-based visualization stack that handles 500,000+ nodes with smooth 60fps animations, multi-repo layered views, and a fully interactive UI.

### 14.1 — Renderer Decision Matrix

| Requirement | Why pyvis fails | Solution chosen |
|---|---|---|
| 50,000+ nodes | Freezes at ~500 | **Sigma.js v3** — WebGL canvas, GPU-accelerated |
| Smooth animation | DOM-based, no RAF loop | **Graphology** + **ForceAtlas2 WebWorker** — off main thread |
| Multi-repo layered views | Single flat graph only | **React** component tree — each repo is an isolated layer |
| Clickable detail panels | Basic tooltip only | **React** side panel with full node metadata |
| Real-time filtering | Full re-render on filter | **Graphology** subgraph slicing — O(1) filter, no re-layout |
| Path highlighting | Not supported | **Sigma.js** programmatic edge/node color override mid-render |
| Performance heat map | Not supported | **Sigma.js** node color reducer driven by BlazeMeter report data |
| Large graph export | Not supported | **html-to-image** + **file-saver** |
| Dark/light themes | Hardcoded CSS | **CSS custom properties** + **React context** |
| Cross-repo edge animation | Not possible | **Sigma.js** animated edges with custom programs |

### 14.2 — Technology Stack

```
ptae-ui/                         ← Self-contained Vite + React app
├── package.json
├── vite.config.ts
├── index.html
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── store/
    │   ├── graphStore.ts        ← Zustand store — graph state
    │   ├── reportStore.ts       ← Zustand store — BlazeMeter report overlay
    │   └── uiStore.ts           ← Zustand store — UI state (filters, theme)
    ├── graph/
    │   ├── graphology.ts        ← Build Graphology MultiDirectedGraph from PTAE JSON
    │   ├── layouts.ts           ← ForceAtlas2 / hierarchical / radial layouts
    │   ├── filters.ts           ← Subgraph slicing
    │   └── overlays.ts          ← Heat map / coverage overlays
    ├── components/
    │   ├── GraphCanvas.tsx      ← Main Sigma.js WebGL canvas
    │   ├── NodeDetailPanel.tsx  ← Slide-in panel on node click
    │   ├── FilterBar.tsx        ← Filter ribbon (type, repo, role)
    │   ├── SearchBar.tsx        ← Fuzzy symbol search
    │   ├── MultiRepoView.tsx    ← Hierarchical upstream/downstream
    │   ├── ApiInventoryView.tsx ← Endpoints colored by HTTP method
    │   ├── E2EFlowView.tsx      ← Sequential animated flow
    │   ├── HeatMapView.tsx      ← p95 vs SLA gradient
    │   ├── CoverageView.tsx     ← JMX coverage indicator
    │   ├── MiniMap.tsx
    │   ├── LegendPanel.tsx
    │   └── ExportButton.tsx
    └── types/
        └── graph.d.ts           ← TypeScript types mirroring Python schema
```

**`package.json` (pin these exact versions):**
```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "graphology": "^0.25.4",
    "graphology-layout": "^0.6.1",
    "graphology-layout-forceatlas2": "^0.10.1",
    "graphology-shortest-path": "^2.1.0",
    "graphology-communities-louvain": "^2.0.0",
    "@react-sigma/core": "^4.2.1",
    "sigma": "^3.0.0",
    "zustand": "^4.5.2",
    "framer-motion": "^11.2.0",
    "html-to-image": "^1.11.11",
    "file-saver": "^2.0.5",
    "fuse.js": "^7.0.0",
    "d3-scale-chromatic": "^3.1.0",
    "lucide-react": "^0.383.0"
  },
  "devDependencies": {
    "vite": "^5.2.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.4.5"
  }
}
```

### 14.3 — Critical Performance Principles

1. **Graphology as data layer, Sigma as render layer** — never pass raw JSON to Sigma; always build Graphology graph first.
2. **ForceAtlas2 MUST run in a WebWorker** — never block the main thread.
3. **`hideEdgesOnMove: true`** — single biggest perf win for pan/zoom.
4. **`sigma.refresh({ skipIndexation: true })`** for color/visibility updates — 10× faster than full refresh.
5. **Filter via `nodeReducer`/`edgeReducer`** — never rebuild graph or re-run layout.
6. **`react-virtual`** for all lists in detail panel — never render > 100 DOM nodes at once.
7. **Lazy-load graph JSON** via streaming fetch — show partial graph while rest loads.

### 14.4 — Performance Benchmarks UI Must Meet

| Scenario | Node Count | Edge Count | Target FPS | Target Load Time |
|---|---|---|---|---|
| Single small repo | < 500 | < 2,000 | 60fps | < 1s |
| Single medium repo | 5,000 | 25,000 | 60fps | < 3s |
| Single large repo | 50,000 | 200,000 | 60fps (pan/zoom) | < 8s |
| Multi-repo platform | 20,000 | 80,000 | 60fps | < 5s |
| Enterprise monorepo | 200,000 | 1,000,000 | 30fps min | < 20s |
| While filtering | any | any | No frame drop | < 100ms response |
| Node click → panel | any | any | Instant | < 50ms |

### 14.5 — Performance Heat Map Integration

A defining PTAE-specific feature: overlay BlazeMeter report data on the graph.

```typescript
// store/reportStore.ts
interface ReportOverlay {
  master_id: number;
  metrics_by_endpoint: Record<string, {
    avg_ms: number;
    p95_ms: number;
    p99_ms: number;
    error_rate_pct: number;
    sla_p95_ms: number;        // From perf-sla.yaml
    sla_status: "pass" | "warn" | "fail";
  }>;
}

// Color logic:
//   green:  current_p95 < sla * 0.7
//   yellow: current_p95 < sla
//   orange: current_p95 < sla * 1.2
//   red:    current_p95 >= sla * 1.2
```

When the heat map view is active, every `API_ENDPOINT` node's color is overridden by its current p95 vs SLA status. Hovering a node shows current/previous p95 side-by-side. Clicking opens the BlazeMeter report URL in a new tab.

### 14.6 — Python ↔ UI Data Bridge

Python backend (`ptae/visualization/json_exporter.py`) exports graph + (optional) BlazeMeter overlay as a single bundle:

```python
def export_graph_json(
    engine: GraphQueryEngine,
    output_path: Path,
    repo_ids: list[str] = None,
    include_cfg_nodes: bool = False,
    bzm_master_id: Optional[int] = None,    # If set, overlay report data
    max_nodes: int = None,
) -> None:
    """Exports unified JSON consumed by React UI."""
    payload = {
        "meta": {...},
        "repos": engine.get_all_repos(repo_ids),
        "nodes": [_serialize_node(n) for n in nodes],
        "edges": [_serialize_edge(e) for e in edges],
        "e2e_flows": engine.get_e2e_flows(),
        "performance_overlay": None,
    }
    if bzm_master_id:
        from ptae.blazemeter.api_client import BlazeMeterClient
        client = BlazeMeterClient(...)
        report = await client.get_aggregate_report(bzm_master_id)
        payload["performance_overlay"] = _build_overlay(report)
    output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
```

### 14.7 — UI Build & Embed Pipeline

The React app is built once and bundled as Python package data, so `pip install perf-testing-automation-engine` delivers the full UI with zero npm install required at end-user time. CLI command `ptae graph --open` builds (if not cached), serves on `localhost:7473`, opens browser.

---

## ═══════════════════════════════════════════════════════
## SECTION 15 — TESTING PTAE ITSELF
## ═══════════════════════════════════════════════════════

PTAE must maintain its own test suite covering:

- **Unit tests:** Each parser tested against a fixture file per language. Each builder tested with hand-crafted graph fixture. Each MCP tool tested with mock query engine.
- **JMX validity tests:** Every generated JMX is loaded into JMeter via `jmeter --validate` in CI and must pass.
- **JMX semantic tests:** Generated JMX is parsed back and asserted to contain expected samplers/assertions per scenario.
- **BlazeMeter integration tests:** Use a dedicated test BlazeMeter project; nightly CI run uploads → runs → compares against a known baseline.
- **Comparison engine tests:** Hand-crafted "previous" and "current" reports with known deltas → assert comparison output matches expected verdict.
- **SLA validation tests:** Verify violations are correctly detected at each severity tier.
- **Edge case regression tests:** Every bullet in Section 11 has a corresponding test case in `tests/edge_cases/`.
- **Performance benchmarks:** Index the `cpython` standard library (> 500 files). Incremental re-index after touching one file < 3 seconds. JMX generation for 100 endpoints < 5 seconds.

---

*End of PROMPT — perf-testing-automation-engine (PTAE) Complete Specification*
*Version: 1.0.0 | Focus: Codebase Graphification → JMX Generation → BlazeMeter Orchestration*
*Compiled from research on: codebase graphification frameworks (graphify, graphiti, codebadger, Joern), JMeter test plan structure, BlazeMeter REST API v4, Bitbucket Cloud/Server APIs, Sigma.js v3, Graphology, react-sigma, ForceAtlas2, and modern performance testing best practices.*
