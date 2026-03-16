Sapiens Decision MCP
Role: Senior Python Backend Engineer & Systems Architect.
Objective: Develop a robust Model Context Protocol (MCP) server in Python using FastMCP that acts as a bridge to the Sapiens Decision Management Suite (REST API v1).

1. Architectural Constraints
Framework: Use fastmcp for the server implementation.

Networking: Use httpx (Asynchronous) for all API communication.

Data Integrity: Use Pydantic v2 for all request/response schemas to ensure strong typing and IDE/Copilot autocompletion.

Environment: Support .env configuration for SAPIENS_BASE_URL, SAPIENS_USER, SAPIENS_PASSWORD, and AUTH_TYPE (USER/SSO).

Transport: Default to stdio for integration with Claude Desktop/GitHub Copilot.

2. Core Functional Requirements
Implement the following "Fat" toolset:

Rule Hierarchy Crawler: A recursive tool that maps the entire dependency tree (Parent/Child/Sub-rules) starting from a Rule Name.

Glossary Introspection: A tool that resolves the Business Glossary constraints for every input fact in a rule's hierarchy (data types, enumerations, and ranges).

Test Package Generator: A composite tool that returns:

A valid JSON payload for rule execution.

A Markdown Table (Tabular format) for business review.

A Logic Validation Report checking for gaps/overlaps in the rule logic.

Explainable Execution: An execution tool that triggers the rule and immediately fetches the natural language "Explanation Trace" to describe why a decision was made.

Impact Analysis: A reverse-lookup tool that finds every rule in the ecosystem currently using a specific Fact Type.

Version Diffing: A tool to compare two versions of a rule to identify logic changes.

3. Data Schema Specifications
Define Pydantic models for:

FactConstraint: (name, datatype, allowed_values, range, sample_value).

RuleNode: A recursive model for the dependency tree.

TestPackage: Containing rule name, JSON, table rows, and hierarchy.

ExecutionResult: Including conclusion, execution_id, and trace_id.

4. Code Style & Copilot Optimization
Docstrings: Provide exhaustive docstrings for every @mcp.tool() to ensure the AI assistant understands exactly when and how to call the tool.

Type Hinting: Use Literal and Union types to restrict valid inputs for severity levels and status codes.

Async/Await: All bridge calls and tool definitions must be non-blocking.

Error Handling: Implement a centralized SapiensBridge class with automatic token refresh and robust error handling for 401/404/500 status codes.

How to use this prompt:
GitHub Copilot (VS Code): Open the Copilot Chat and paste the prompt above.

Claude/Gemini: Paste this prompt into a fresh chat and ask it to: "Generate the server.py based on these technical specs."

Modular Expansion: Because the prompt defines the Bridge and Data Models first, you can easily ask the AI to "Add a 7th tool for batch CSV processing" later, and it will maintain the same code pattern.


Use below code for the reference 


import os
import asyncio
import httpx
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Literal
from pydantic import BaseModel, Field
from fastmcp import FastMCP
from dotenv import load_dotenv

# --- 1. SETUP & CONFIG ---
load_dotenv()
SAPIENS_BASE_URL = os.getenv("SAPIENS_BASE_URL", "https://api.sapiensdecision.com/v1")
AUTH_TYPE = os.getenv("AUTH_TYPE", "USER") 

mcp = FastMCP("Sapiens-Ultimate-Bridge")

# --- 2. DATA MODELS (COPILOT INTELLIGENCE) ---

class FactConstraint(BaseModel):
    name: str
    datatype: Literal["String", "Number", "Boolean", "Date", "DateTime", "Collection"]
    allowed_values: Optional[List[Any]] = None
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    is_required: bool = True
    description: str = ""

class RuleNode(BaseModel):
    name: str
    rule_id: str
    type: str
    version: str
    status: str
    children: List['RuleNode'] = []

RuleNode.model_rebuild()

class ValidationIssue(BaseModel):
    severity: Literal["Error", "Warning", "Info"]
    message: str
    rule_component: str # e.g., "Condition Column 3"

class TestPackage(BaseModel):
    rule_name: str
    json_payload: Dict[str, Any]
    tabular_data: List[Dict[str, Any]]
    hierarchy: RuleNode
    validation_report: List[ValidationIssue] = []

class ExecutionResult(BaseModel):
    execution_id: str
    conclusion: Dict[str, Any]
    trace_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# --- 3. API CORE ---

class SapiensBridge:
    def __init__(self):
        self.client = httpx.AsyncClient(base_url=SAPIENS_BASE_URL, timeout=90.0)
        self.token: Optional[str] = None

    async def _auth(self):
        if AUTH_TYPE == "USER":
            resp = await self.client.post("/auth/login", json={
                "username": os.getenv("SAPIENS_USER"), "password": os.getenv("SAPIENS_PASSWORD")
            })
        else:
            resp = await self.client.post("/auth/sso", headers={"Authorization": f"Bearer {os.getenv('SSO_TOKEN')}"})
        resp.raise_for_status()
        self.token = resp.json().get("access_token")

    async def call(self, method: str, path: str, **kwargs) -> Any:
        if not self.token: await self._auth()
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self.token}"
        response = await self.client.request(method, path, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()

api = SapiensBridge()

# --- 4. THE ALL-IN-ONE TOOLSET ---

@mcp.tool()
async def analyze_full_rule_ecosystem(rule_name: str) -> TestPackage:
    """
    EXTREME USE CASE: Performs a deep-dive on a rule.
    1. Crawls hierarchy (Parent/Child/Sub-rules)
    2. Maps all Glossary constraints
    3. Generates valid JSON & Table test data
    4. Runs logic validation to find gaps/overlaps.
    """
    # 1. Resolve Hierarchy
    async def get_tree(name: str):
        data = await api.call("GET", f"/models/{name}/details")
        node = RuleNode(name=name, rule_id=data['id'], type=data['type'], version=data['v'], status=data['status'])
        node.children = await asyncio.gather(*[get_tree(c['name']) for c in data.get("dependencies", [])])
        return node
    
    tree = await get_tree(rule_name)
    
    # 2. Aggregated Inputs & Validation
    inputs = await api.call("GET", f"/models/{rule_name}/aggregated-inputs")
    validation = await api.call("GET", f"/models/{rule_name}/validate")
    
    # 3. Format Data
    payload = {i['name']: (i.get('sample') or "Sample") for i in inputs}
    table = [{"Fact": i['name'], "Type": i['datatype'], "Sample": payload[i['name']]} for i in inputs]
    
    return TestPackage(
        rule_name=rule_name,
        json_payload=payload,
        tabular_data=table,
        hierarchy=tree,
        validation_report=[ValidationIssue(**v) for v in validation.get("issues", [])]
    )

@mcp.tool()
async def execute_and_explain(rule_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a rule and immediately fetches the explanation trace.
    Returns: { "conclusion": ..., "why": "Explanation of triggered logic" }
    """
    exec_data = await api.call("POST", "/execute", json={"modelName": rule_name, "data": input_data})
    trace = await api.call("GET", f"/trace/{exec_data['execution_id']}/summary")
    return {
        "result": exec_data['conclusion'],
        "explanation": trace.get("natural_language_summary"),
        "execution_id": exec_data['execution_id']
    }

@mcp.tool()
async def search_glossary_and_usage(query: str) -> List[Dict[str, Any]]:
    """
    Finds a Fact Type and lists EVERY rule that uses it.
    Critical for 'Impact Analysis' when changing a business term.
    """
    return await api.call("GET", "/glossary/usage-search", params={"q": query})

@mcp.tool()
async def batch_test_from_local(rule_name: str, file_path: str) -> List[ExecutionResult]:
    """
    Reads a local JSON file containing multiple test cases and 
    runs them all against the Sapiens engine in parallel.
    """
    with open(file_path, 'r') as f:
        test_cases = json.load(f) # Expects a list of dicts
    
    tasks = [execute_and_explain(rule_name, case) for case in test_cases]
    results = await asyncio.gather(*tasks)
    return [ExecutionResult(execution_id=r['execution_id'], conclusion=r['result']) for r in results]

@mcp.tool()
async def compare_rule_versions(rule_name: str, v1: str, v2: str) -> Dict[str, Any]:
    """
    Performs a 'Diff' between two versions of the same rule.
    Shows added/removed conditions or changes in conclusion values.
    """
    return await api.call("GET", f"/models/{rule_name}/diff", params={"v1": v1, "v2": v2})

if __name__ == "__main__":
    mcp.run()
