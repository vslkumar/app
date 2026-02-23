**Role:** Act as an expert Python Backend Developer and Model Context Protocol (MCP) Architect.

**Objective:** Create a production-ready MCP server using Python and the official `mcp` Python SDK. This server allows an AI assistant to securely SSH into remote servers, extract logs asynchronously, and actively perform Root Cause Analysis (RCA) using highly flexible, parameterized log-investigation tools.

**Core Requirements:**
1. **Inputs/Configuration:** The system must manage `host`, `port`, `username`, `applicationName`, `logpath` (can be a directory or list of files), and `cluster` (e.g., dev, staging, prod).
2. **MCP Resources (The Environment Map):**
   - Implement an MCP `Resource` endpoint: `config://{cluster}/{applicationName}`.
   - Returns a JSON object with connection details (`host`, `port`, `username`, standard `logpath`), allowing the LLM to understand the topology before investigating. **Never include passwords in this JSON response.**

3. **MCP Tools (The RCA & Omnipotent Grep Toolkit):** Implement the following distinct tools using `pydantic` for strict schema validation. The Python server must dynamically build the `grep` commands using these parameters to allow advanced searching without exposing raw shell access.
   - **`advanced_log_grep` (The Core Engine):** - Arguments: `cluster` (str), `applicationName` (str), `search_pattern` (str), `use_regex` (bool, default: False), `ignore_case` (bool, default: True), `invert_match` (bool, default: False), `before_context` (int, default: 0), `after_context` (int, default: 0), `max_count` (int, default: 1000).
     - Logic: Constructs a safe grep command. If `use_regex`, use `grep -E`. If `ignore_case`, use `grep -i`. If `invert_match`, use `grep -v`. Apply `-B` and `-A` for context, and `-m` for max count. **Always sanitize `search_pattern` using `shlex.quote()` to prevent command injection.**
   - **`get_error_context` (Quick RCA Tool):** - Arguments: `cluster`, `applicationName`, `timestamp_or_id` (str), `context_lines` (int, default: 50).
     - Logic: Uses `grep -C {context_lines}` to pull the logs immediately preceding and following a specific crash/timestamp.
   - **`search_correlated_logs` (Distributed Tracing Tool):**
     - Arguments: `cluster`, `applicationName`, `trace_id` (str).
     - Logic: Searches across all configured log files in the `logpath` for a specific distributed Trace ID or Request ID.

**Additional Python/RCA Optimizations Required:**
1. **Secure Password & Credential Management:**
   - Support password-based and key-based SSH authentication.
   - Read passwords dynamically from environment variables using `python-dotenv` (e.g., `os.environ.get(f"SSH_PASSWORD_{cluster.upper()}")`). 
   - **Crucial:** Ensure passwords are never logged to the console, never hardcoded in the `config.json`, and never returned in any MCP Resource or Tool response.
2. **Asynchronous Connection Pooling:** Strictly use the `asyncssh` and `asyncio` libraries. Maintain an active SSH connection pool with an idle timeout so the LLM can rapidly fire multiple queries without re-authenticating every time.
3. **JSON Log Parsing & Minimization:** If logs are JSON, use Python's `json` module to strip out noisy, irrelevant fields before returning the payload, preserving the LLM's context window.
4. **Smart Truncation & Summarization:** Truncate massive stack traces in the middle. Keep the top (the exception) and the bottom (the root cause), adding `\n[...Stack trace truncated for LLM context...]\n`. Ensure tool outputs stay under a strict 20,000 character limit.

**Output Structure:**
- Provide a `requirements.txt` including `mcp`, `asyncssh`, `pydantic`, and `python-dotenv`.
- Provide the well-commented Python code for `server.py` implementing the FastMCP or standard MCP server routing.
- Provide a sample `config.json` (containing hosts, usernames, and log paths) and a `.env.example` file (showing how to inject the passwords).
- Provide a `README.md` focusing on how to configure the server securely for an MCP client like Claude Desktop or GitHub Copilot.
