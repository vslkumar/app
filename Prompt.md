
# IBM DB2 Connector MCP Server — Full Replication Prompt

> **Purpose**: Copy-paste this entire prompt into any AI coding agent (Claude, GPT, Copilot, Gemini, etc.) to recreate the `db2-connector-mcp` server from scratch. Every detail is included — no prior context needed.

---

## THE PROMPT

```
Create an MCP (Model Context Protocol) server in Python called "db2-connector-mcp" that connects to IBM DB2 databases, discovers table foreign key hierarchies, and generates the correct deletion order from child tables to parent tables. The server must handle 100+ tables with multi-level hierarchies, detect circular dependencies, and work in restricted DB2 environments where only SYSIBM catalog tables are available (NOT SYSCAT views).

## PROJECT STRUCTURE

Create the following files:

```
db2-connector-mcp/
├── server.py           # Main MCP server (all logic in one file)
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment variables
├── .gitignore          # Git ignore rules
└── README.md           # Documentation
```

## REQUIREMENTS.TXT

```
mcp>=1.0.0
ibm_db>=3.2.0
python-dotenv>=1.0.0
```

## .ENV.EXAMPLE

```
DB2_HOST=localhost
DB2_PORT=50000
DB2_DATABASE=SAMPLE
DB2_USER=db2admin
DB2_PASSWORD=your_password_here
```

## .GITIGNORE

```
.env
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
dist/
build/
```

## TECHNOLOGY & FRAMEWORK

- **Language**: Python 3.10+ (required by MCP SDK)
- **MCP SDK**: Use `mcp` package with `FastMCP` from `mcp.server.fastmcp`
- **DB2 Driver**: Use `ibm_db` (NOT SQLAlchemy, NOT ibm_db_sa — use the raw ibm_db driver directly)
- **Transport**: stdio (default for MCP)
- **Environment**: `python-dotenv` for loading `.env` files

## CRITICAL CONSTRAINT: USE SYSIBM, NOT SYSCAT

The target DB2 environment does NOT have access to `SYSCAT` views. Only these system schemas are available:
- SYSIBM
- SYSTOOLS
- SYSDBA
- SYSXDB
- SYSCCEL
- SYSIBMTS

Therefore, ALL catalog queries MUST use `SYSIBM` base tables instead of `SYSCAT` views.

### Column Name Mapping (SYSCAT → SYSIBM)

This is the most important detail. The SYSIBM tables use DIFFERENT column names than SYSCAT views:

**Schema Discovery** (no SYSCAT.SCHEMATA available):
- Derive schemas from: `SELECT DISTINCT CREATOR FROM SYSIBM.SYSTABLES`
- Column: `CREATOR` (not `SCHEMANAME`)

**Tables** (SYSIBM.SYSTABLES instead of SYSCAT.TABLES):
| SYSCAT Column | SYSIBM Column | Description |
|---------------|---------------|-------------|
| TABSCHEMA     | CREATOR       | Schema name |
| TABNAME       | NAME          | Table name  |
| CARD          | CARDF         | Row count estimate |
| TYPE          | TYPE          | 'T' for table (same) |

**Columns** (SYSIBM.SYSCOLUMNS instead of SYSCAT.COLUMNS):
| SYSCAT Column | SYSIBM Column | Description |
|---------------|---------------|-------------|
| TABSCHEMA     | TBCREATOR     | Schema name |
| TABNAME       | TBNAME        | Table name  |
| COLNAME       | NAME          | Column name |
| TYPENAME      | COLTYPE       | Data type   |
| LENGTH        | LENGTH        | Same        |
| SCALE         | SCALE         | Same        |
| NULLS         | NULLS         | 'Y' or 'N' (same) |
| KEYSEQ        | KEYSEQ        | PK sequence (same) |
| DEFAULT       | DEFAULT       | Default value (same) |
| COLNO         | COLNO         | Column position (same) |

**Foreign Key Relationships** (SYSIBM.SYSRELS instead of SYSCAT.REFERENCES):
| SYSCAT Column  | SYSIBM Column  | Description |
|----------------|----------------|-------------|
| TABSCHEMA      | CREATOR        | Child table schema |
| TABNAME        | TBNAME         | Child table name |
| CONSTNAME      | RELNAME        | FK constraint name |
| REFTABSCHEMA   | REFTBCREATOR   | Parent table schema |
| REFTABNAME     | REFTBNAME      | Parent table name |
| DELETERULE     | DELETERULE     | Delete rule: A/C/N/R (same) |
| UPDATERULE     | UPDATERULE     | Update rule: A/R (same) |
| FK_COLNAMES    | ❌ NOT AVAILABLE | Use SYSIBM.SYSFOREIGNKEYS instead |
| PK_COLNAMES    | ❌ NOT AVAILABLE | Use SYSIBM.SYSFOREIGNKEYS instead |

**FK Column Details** (SYSIBM.SYSFOREIGNKEYS — no SYSCAT equivalent needed):
- Columns: CREATOR, RELNAME, COLNAME, COLSEQ, COLNO
- Join with SYSIBM.SYSRELS on: FK.CREATOR = R.CREATOR AND FK.RELNAME = R.RELNAME

### Delete Rule Codes
- `A` = NO ACTION
- `C` = CASCADE
- `N` = SET NULL
- `R` = RESTRICT

### Update Rule Codes
- `A` = NO ACTION
- `R` = RESTRICT

## SERVER ARCHITECTURE

### Module-Level Connection Store
Use module-level dictionaries to store DB2 connections so tools can share them:
```python
_connections: dict[str, object] = {}      # alias -> ibm_db connection handle
_connection_info: dict[str, dict] = {}    # alias -> metadata
```

### Helper: _get_conn(alias)
Retrieve a connection by alias. Raise RuntimeError if not connected.

### Helper: _execute_query(conn, sql, params)
Execute parameterized SQL using ibm_db:
1. `ibm_db.prepare(conn, sql)`
2. For each param: `ibm_db.bind_param(stmt, idx, val)` — idx is 1-based
3. `ibm_db.execute(stmt)`
4. Loop with `ibm_db.fetch_assoc(stmt)` to collect rows
5. **IMPORTANT**: ibm_db returns uppercase keys with trailing spaces. Normalize all keys to lowercase and strip all string values: `{k.lower().strip(): v.strip() if isinstance(v, str) else v for k, v in result.items()}`

### FastMCP Setup
```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("DB2 Connector MCP", json_response=True)
```

### Entry Point
```python
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## 12 MCP TOOLS TO IMPLEMENT

All tools are decorated with `@mcp.tool()` and return `dict`.

### Tool 1: connect_db2
- **Params**: host (str), port (int), database (str), user (str), password (str), alias (str = "default")
- **Logic**: Build DSN string: `DATABASE={database};HOSTNAME={host};PORT={port};PROTOCOL=TCPIP;UID={user};PWD={password};`
- Call `ibm_db.connect(dsn, "", "")`
- Store connection in `_connections[alias]`
- Get server info via `ibm_db.server_info(conn)` — use `getattr(server_info, "DBMS_VER", "unknown")`
- **Returns**: `{status, alias, database, host, db_server_version, db_product_name}`
- Wrap in try/except, return error dict on failure

### Tool 2: disconnect_db2
- **Params**: alias (str = "default")
- **Logic**: Pop from `_connections`, call `ibm_db.close(conn)`
- **Returns**: `{status, alias}`

### Tool 3: list_schemas
- **SQL**: `SELECT DISTINCT CREATOR FROM SYSIBM.SYSTABLES WHERE CREATOR NOT LIKE 'SYS%' AND CREATOR NOT IN ('NULLID', 'SQLJ') ORDER BY CREATOR`
- **Returns**: `{schemas: [...], count: N}`

### Tool 4: list_tables
- **Params**: schema (str), alias (str = "default")
- **SQL**: `SELECT NAME, CARDF AS ROW_COUNT, TYPE FROM SYSIBM.SYSTABLES WHERE CREATOR = ? AND TYPE = 'T' ORDER BY NAME`
- Always `.upper()` the schema parameter before binding
- **Returns**: `{schema, tables: [{table_name, row_count, type}], count}`

### Tool 5: get_table_details
- **Params**: schema (str), table (str), alias (str = "default")
- **SQL**: `SELECT C.NAME, C.COLTYPE, C.LENGTH, C.SCALE, C.NULLS, C.KEYSEQ, C.DEFAULT FROM SYSIBM.SYSCOLUMNS C WHERE C.TBCREATOR = ? AND C.TBNAME = ? ORDER BY C.COLNO`
- **Returns**: `{schema, table, columns: [{column_name, type, length, scale, nullable (bool: NULLS == 'Y'), primary_key_seq, default_value}], column_count}`

### Tool 6: get_foreign_keys
- **Params**: schema (str), table (Optional[str] = None), alias (str = "default")
- **If table provided**: Query SYSIBM.SYSRELS WHERE CREATOR = ? AND TBNAME = ?
- **If table omitted**: Query SYSIBM.SYSRELS WHERE CREATOR = ? OR REFTBCREATOR = ? (same schema for both params)
- **Select columns**: R.RELNAME AS FK_NAME, R.CREATOR AS CHILD_SCHEMA, R.TBNAME AS CHILD_TABLE, R.REFTBCREATOR AS PARENT_SCHEMA, R.REFTBNAME AS PARENT_TABLE, R.DELETERULE, R.UPDATERULE
- **For each FK**: Query SYSIBM.SYSFOREIGNKEYS to get column names: `SELECT FK.COLNAME, FK.COLSEQ FROM SYSIBM.SYSFOREIGNKEYS FK WHERE FK.CREATOR = ? AND FK.RELNAME = ? ORDER BY FK.COLSEQ`
- Join column names with ", "
- Map DELETERULE: A→NO ACTION, C→CASCADE, N→SET NULL, R→RESTRICT
- Map UPDATERULE: A→NO ACTION, R→RESTRICT
- **Returns**: `{foreign_keys: [{fk_name, child_schema, child_table, parent_schema, parent_table, delete_rule, update_rule, fk_columns}], count}`

### Internal Function: _build_hierarchy_graph(conn, schema)
This is the core graph builder used by tools 7-12.
- **Query SYSIBM.SYSRELS**: `SELECT R.TBNAME AS CHILD_TABLE, R.REFTBNAME AS PARENT_TABLE, R.RELNAME AS FK_NAME, R.DELETERULE FROM SYSIBM.SYSRELS R WHERE R.CREATOR = ? AND R.REFTBCREATOR = ? ORDER BY R.TBNAME`
- Build:
  - `children`: dict — parent_table → [child_table, ...] (defaultdict(list))
  - `parents`: dict — child_table → [parent_table, ...] (defaultdict(list))
  - `all_tables`: set of all table names
  - `edges`: list of tuples (child, parent)
  - `fk_details`: raw row data for constraint names
- **Also query all tables** to include isolated ones (no FK): `SELECT NAME FROM SYSIBM.SYSTABLES WHERE CREATOR = ? AND TYPE = 'T'` — add all to all_tables set
- **Returns**: dict with keys: children, parents, all_tables, edges, fk_details

### Internal Function: _topological_sort_delete_order(graph)
Uses **Kahn's algorithm** for topological sort.
- **Return type**: `tuple[list[str], list[str]]` — (ordered_tables, circular_tables)
- **Graph semantics**: edge (child, parent) means child depends on parent. For deletion, child must come before parent.
- **In-degree**: `in_degree[parent] += 1` for each (child, parent) edge. Tables with in_degree=0 are leaf children (nothing depends on them) → delete first.
- **Algorithm**:
  1. Initialize `in_degree: dict[str, int] = {table: 0 for table in all_tables}`
  2. Initialize `adjacency: dict[str, list[str]] = {table: [] for table in all_tables}`
  3. For each (child, parent): adjacency[child].append(parent), in_degree[parent] += 1
  4. Seed queue with all tables where in_degree == 0
  5. Process queue: pop table, add to order, for each neighbor in adjacency[table]: decrement in_degree, enqueue if 0
  6. If len(order) != len(all_tables): circular dep detected. Return (order, list(remaining_tables))
  7. Otherwise: return (order, [])

### Internal Function: _get_hierarchy_levels(graph)
Assigns depth level to each table recursively:
- Level 0 = root tables (no parents)
- Level N = max(parent_levels) + 1
- Handle circular refs by checking visited set, assign level 0 to break cycle
- Group by level using defaultdict(list)
- **Returns**: dict[int, list[str]]

### Tool 7: get_table_hierarchy
- **Params**: schema (str), alias (str = "default")
- Call _build_hierarchy_graph and _get_hierarchy_levels
- Identify root tables (not in parents dict), leaf tables (not in children dict), isolated (in neither)
- **Returns**: `{schema, total_tables, total_fk_relationships, hierarchy_levels: {"0": [...], "1": [...]}, max_depth, root_tables, leaf_tables, isolated_tables, relationships: [{child, parent}]}`
- Convert level keys to strings in output

### Tool 8: generate_delete_order
- **Params**: schema (str), tables (Optional[str] = None — comma-separated), alias (str = "default")
- If tables provided: split by comma, strip, upper. Filter graph to only include those tables and edges between them
- Call _topological_sort_delete_order
- **Returns**: `{schema, delete_order: [...], total_tables, explanation: "Delete from tables in this order (top to bottom)..."}`
- If circular: add circular_dependencies and warning fields

### Tool 9: generate_delete_sql
- **Params**: schema (str), where_clause (Optional[str] = None), tables (Optional[str] = None), include_disable_constraints (bool = False), alias (str = "default")
- Same table filtering as Tool 8
- Build list of statement dicts, each with: step (int), type (str), sql (str), description (str)
- **If include_disable_constraints**:
  - Step 1: DISABLE — for each table in order that has FKs, generate: `ALTER TABLE "SCHEMA"."TABLE" ALTER FOREIGN KEY FK_NAME NOT ENFORCED;`
  - Last step: ENABLE — same but `ENFORCED`
- **DELETE statements**: For each table in order:
  - With where: `DELETE FROM "SCHEMA"."TABLE" WHERE {where_clause};`
  - Without: `DELETE FROM "SCHEMA"."TABLE";`
- Compose full_script by joining all SQL with double newlines
- **Returns**: `{schema, statements: [...], total_steps, full_script}`

### Tool 10: visualize_hierarchy
- **Params**: schema (str), alias (str = "default")
- **Text tree**: Build ASCII art tree starting from root tables using recursive _draw_tree function
  - Use └── and ├── connectors, │ for continuation
  - Detect circular refs (visited set), show ↺ marker
  - Show 📦 emoji for root tables, 📋 for isolated tables
  - Include level summary at bottom
- **Mermaid diagram**: Generate `graph TD` with edges: `PARENT[PARENT] --> CHILD[CHILD]`
- **Returns**: `{text_tree: "...", mermaid_diagram: "..."}`

### Tool 11: analyze_delete_impact
- **Params**: schema (str), table (str), alias (str = "default")
- **BFS descendants** (tables that depend on target): walk children dict
- **BFS ancestors** (tables target depends on): walk parents dict
- **Row counts**: For each affected table, run `SELECT COUNT(*) AS CNT FROM "SCHEMA"."TABLE"` — catch exceptions, return "error"
- Build subset graph of target + affected tables, run topological sort for safe delete order
- **Returns**: `{target_table, schema, tables_that_depend_on_target, tables_target_depends_on, row_counts, safe_delete_order, total_affected_tables, circular_dependencies, recommendation}`

### Tool 12: find_relationship_path
- **Params**: schema (str), from_table (str), to_table (str), alias (str = "default")
- **BFS undirected**: Consider both parent and child edges as neighbors
- Find shortest path(s) between the two tables
- **Annotate each edge**: Check if (a,b) is in edges → "a is child of b"; if (b,a) → "a is parent of b"
- **Returns**: `{from_table, to_table, path_exists (bool), paths: [{tables, edges, depth}], shortest_distance}` or `{path_exists: False, message: "..."}`

## SETUP INSTRUCTIONS TO INCLUDE IN README

```bash
# Create virtual environment (requires Python 3.10+)
# If only Python 3.9 available, install uv first:
# curl -LsSf https://astral.sh/uv/install.sh | sh
# Then: uv venv --python 3.12 .venv

python -m venv .venv
source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt

# Configure
cp .env.example .env         # Edit with your DB2 credentials

# Run
python server.py
```

## MCP CLIENT CONFIGURATION

For Claude Desktop / Cline / Antigravity / any MCP client, add to the MCP server config JSON:

```json
{
  "mcpServers": {
    "db2-connector": {
      "command": "/absolute/path/to/db2-connector-mcp/.venv/bin/python",
      "args": ["/absolute/path/to/db2-connector-mcp/server.py"]
    }
  }
}
```

## IMPORTANT IMPLEMENTATION NOTES

1. **Import ibm_db locally** inside each tool function (not at module top level) — this prevents import errors if ibm_db is not installed and allows the MCP server to load its tool definitions even without the DB2 driver
2. **Always .upper() schema and table names** before passing to SQL parameters — DB2 catalog stores names in uppercase
3. **ibm_db.fetch_assoc returns uppercase keys with trailing spaces** — always normalize: `k.lower().strip()`
4. **ibm_db.fetch_assoc returns string values with trailing spaces** — always strip: `v.strip() if isinstance(v, str) else v`
5. **Use parameterized queries** (? placeholders with ibm_db.bind_param) — never concatenate user input into SQL strings (except for the table names in DELETE generation which are from the catalog)
6. **DSN format**: `DATABASE=...;HOSTNAME=...;PORT=...;PROTOCOL=TCPIP;UID=...;PWD=...;`
7. **Connection handling**: Store in module-level dict so multiple tools can use the same connection
8. **Multiple connection support**: The alias parameter allows connecting to multiple DB2 instances simultaneously
9. **All tools return dict** not strings — FastMCP with json_response=True handles serialization

Generate all the files now. Put everything in a single server.py file (no splitting into modules).
```

---

> [!TIP]
> **To use this prompt**: Copy everything between the outer ` ``` ` markers and paste it into any AI coding agent. The agent should produce a fully functional MCP server that matches the implementation exactly.
