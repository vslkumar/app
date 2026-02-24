Role & Context:
You are an expert Principal Backend Engineer specializing in Python 3.10+, async programming, and the Model Context Protocol (MCP). Your objective is to build a robust, production-ready, and generic REST API MCP server using the official mcp Python SDK (specifically leveraging the FastMCP class).

Project Overview:
Create a flexible MCP server that allows an AI assistant to interact with advanced REST APIs (such as Elasticsearch, Kibana, or any data-heavy API). The server must connect using dynamically configured Base URLs and Authentication methods. Crucially, it must support highly complex, deeply nested JSON query payloads (e.g., Elasticsearch Query DSL, GraphQL payloads, or complex POST bodies) without failing schema validation.

Technical Stack:

Language: Python 3.10+

SDK: mcp (using from mcp.server.fastmcp import FastMCP)

HTTP Client: httpx (for asynchronous, non-blocking HTTP requests)

Type Hinting/Validation: Standard Python typing (Dict, Any, Optional) for FastMCP tool schema generation.

Configuration & Security:
The server must NEVER hardcode credentials. Use os.environ and python-dotenv to load the following environment variables:

API_BASE_URL (e.g., http://localhost:9200 or https://api.mydomain.com)

API_AUTH_TYPE (e.g., basic, bearer, or none)

API_USERNAME (If using basic auth)

API_PASSWORD (If using basic auth)

API_BEARER_TOKEN (If using bearer auth)

Required MCP Tools to Implement:

execute_rest_request (The Core Query Engine)

Description: Executes an HTTP request against the configured REST API. This tool is designed to handle complex queries, such as Elasticsearch match_phrase, nested aggregations, or bulk operations.

Arguments:

endpoint (str, required): The URL path to append to the Base URL (e.g., /my-index/_search).

method (str, required): The HTTP method (GET, POST, PUT, DELETE). Default to POST.

payload (Dict[str, Any], optional): The raw JSON payload/body. Critical: This must be typed flexibly so the AI can construct deeply nested JSON structures without triggering rigid Pydantic/FastMCP validation errors.

query_params (Dict[str, Any], optional): URL query parameters.

Implementation: Use httpx.AsyncClient to dispatch the request.

fetch_api_schema_or_mapping

Description: Retrieves the schema, index mapping, or OpenAPI specification from the target API.

Arguments: endpoint (str, required) - e.g., /my-index/_mapping.

Purpose: Allows the LLM to inspect the database structure, field names, and data types before attempting to write complex queries.

check_api_health

Description: Pings the root or health endpoint of the API to verify connectivity and authentication status.

Key Architectural Requirements:

Error Reflection: Wrap the httpx calls in try/except httpx.HTTPStatusError. If the API returns a 4xx or 5xx error (e.g., due to a malformed match_phrase query), catch it, parse the API's JSON error response, and return it directly as a string to the LLM. Do not crash the server. The LLM needs this exact error output to self-correct its JSON syntax.

Async/Await Native: Ensure all tool functions defined with @mcp.tool() are async def and use httpx.AsyncClient to prevent blocking the MCP stdio transport layer.

Initialization: Set up the server using mcp = FastMCP("UniversalRESTAdapter"). Start the server using mcp.run(transport="stdio") inside the if __name__ == "__main__": block so it integrates seamlessly with Claude Desktop or GitHub Copilot.

Logging: Configure standard Python logging to output to sys.stderr ONLY. Printing standard output to sys.stdout will corrupt the MCP JSON-RPC protocol and break the server.

Please generate the complete, well-commented server.py and requirements.txt. Include a short README section on how to configure the Claude Desktop/Cursor/Copilot JSON configuration file to run this server via the uv package manager.
