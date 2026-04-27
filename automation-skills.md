---
name: automation-skills
description: >
  Use this skill whenever the user provides an Excel file (.xlsx / .xls) and wants to
  generate or extend Cucumber BDD test automation inside a Spring Boot project.
  Triggers: user uploads an Excel file intended as test input, mentions "generate feature
  file", "create scenarios from Excel", "add test cases from sheet", or asks to map Excel
  tabs to Cucumber scenarios. The skill parses the workbook (file = Feature, each tab =
  Scenario), scans the entire project for existing .feature files and Java/Kotlin Step
  Definition classes, reuses every matching artefact it finds, and only creates new files
  or methods when no match exists. Do NOT use this skill for unrelated Excel tasks, plain
  Cucumber editing, or non-Spring-Boot projects.
compatibility: >
  Spring Boot 2.x / 3.x + Cucumber 6.x / 7.x (Java or Kotlin),
  Maven or Gradle (single- or multi-module), openpyxl, xlrd
---

# Automation Skill — Excel → Cucumber BDD (Spring Boot)

## Purpose

Manually translating test-case spreadsheets into Cucumber `.feature` files and Java/Kotlin
Step Definition classes is error-prone and causes duplicated glue code. This skill encodes
the complete, hardened workflow so that Claude can:

1. **Parse** an uploaded Excel workbook as a structured test specification — handling every
   known Excel edge case before a single line of Gherkin is written.
2. **Discover** every existing `.feature` file, `Scenario`, `Background`, and annotated
   `@Step` method in the project without missing hidden directories or non-standard layouts.
3. **Reuse** every matching artefact exactly once — zero duplication, zero ambiguity.
4. **Generate** only net-new artefacts, wired correctly into the Spring Boot test context.
5. **Validate** all outputs against a hardened checklist before presenting them to the user.

---

## MANDATORY EXECUTION ORDER

```
Phase 0  →  Phase 1  →  Phase 2  →  Phase 3  →  Phase 4
→  Phase 5  →  Phase 6  →  Phase 7  →  Phase 8  →  Phase 9
```

**Never skip or reorder phases.** Each phase gate-checks the previous one.
If any phase produces a BLOCK condition, stop and report to the user before proceeding.

---

## Conceptual Mapping

| Excel concept                         | Cucumber / Spring Boot concept                     |
|---------------------------------------|----------------------------------------------------|
| Workbook (`.xlsx` / `.xls`)           | One `.feature` file                                |
| Workbook filename                     | `Feature:` title                                   |
| Sheet / Tab name                      | `Scenario:` or `Scenario Outline:` title           |
| Row 1 (headers)                       | Column names → Gherkin keyword roles + param names |
| Single data row (row 2)               | Plain `Scenario` with concrete step values         |
| Multiple data rows (row 2+)           | `Scenario Outline` + `Examples:` table             |
| All tabs sharing identical Given cols | `Background:` block extracted (hoisted)            |
| Tab named `BACKGROUND`                | Literal Background block (special handling)        |
| Tab named `TAGS`                      | Feature-level and scenario-level tag metadata      |

---

## Phase 0 — Pre-flight Checks

Before reading or writing anything, confirm all of the following.
Report any **BLOCK** to the user immediately and wait for resolution.

### 0.1 File format check
```bash
file /mnt/user-data/uploads/<input_file>
```

| Result                          | Action                                                          |
|---------------------------------|-----------------------------------------------------------------|
| `Microsoft Excel 2007+` (xlsx)  | Proceed — use `openpyxl`                                       |
| `Microsoft Excel` (xls/97-2003) | Convert first (see 0.1a) or ask user to re-save as `.xlsx`     |
| `Zip archive`                   | Proceed — xlsx files are valid ZIP archives; openpyxl handles  |
| `HTML`                          | BLOCK — some Excel exports are HTML disguised as .xls; ask user for real .xlsx |
| `CSV`                           | Not a workbook; re-interpret as single-tab workbook or ask user |

#### 0.1a — Convert `.xls` → `.xlsx`
```bash
pip install xlrd openpyxl --break-system-packages --quiet
```
```python
import xlrd, openpyxl

def xls_to_xlsx(xls_path: str, out_path: str) -> str:
    rb = xlrd.open_workbook(xls_path)
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for sheet in rb.sheets():
        ws = wb.create_sheet(title=sheet.name)
        for rx in range(sheet.nrows):
            ws.append([sheet.cell_value(rx, cx) for cx in range(sheet.ncols)])
    wb.save(out_path)
    return out_path

converted = xls_to_xlsx("/mnt/user-data/uploads/input.xls", "/tmp/input_converted.xlsx")
```

### 0.2 Password / encryption check
```python
from zipfile import BadZipFile
try:
    import openpyxl
    openpyxl.load_workbook("/mnt/user-data/uploads/<file>", read_only=True)
except BadZipFile:
    raise SystemExit("BLOCK: File may be encrypted or corrupt. Ask user to remove password.")
except Exception as e:
    raise SystemExit(f"BLOCK: Cannot open workbook — {e}")
```

### 0.3 Project layout detection
```bash
# Build tool
ls pom.xml build.gradle build.gradle.kts 2>/dev/null

# Multi-module detection
find . -name "pom.xml" -not -path "*/target/*" | head -20
find . -name "build.gradle*" -not -path "*/build/*" | head -20

# Language detection
find . -name "*.java" -path "*/test/*" | head -5
find . -name "*.kt"   -path "*/test/*" | head -5

# Non-standard test source roots
find . -type d \( -name "it" -o -name "itest" -o -name "integration-test" \) \
     -not -path "*/target/*" -not -path "*/build/*"
```

Record these constants for use in all later phases:
- `BUILD_TOOL` = `maven` | `gradle`
- `LANGUAGE` = `java` | `kotlin` | `mixed`
- `TEST_ROOT` = `src/test/java` (or discovered custom path)
- `RESOURCE_ROOT` = `src/test/resources` (or discovered custom path)
- `IS_MULTI_MODULE` = `true` | `false` (if true, identify the test module)
- `BASE_PACKAGE` = inferred from existing Step Definition `package` declarations

### 0.4 Cucumber version detection
```bash
# Maven
grep -A1 "cucumber" pom.xml | grep "version" | head -5

# Gradle
grep "cucumber" build.gradle build.gradle.kts 2>/dev/null | grep "version" | head -5
```

Record `CUCUMBER_VERSION`. Impacts:
- **Cucumber 6**: `io.cucumber.java.en.*`, `@CucumberOptions`, `@RunWith(Cucumber.class)`
- **Cucumber 7**: `io.cucumber.spring.CucumberContextConfiguration`, `@ConfigurationParameter`, `@Suite`
- **Kotlin**: `io.cucumber.java8.En` (lambda) vs `io.cucumber.kotlin.*` (annotation — preferred in 7+)

---

## Phase 1 — Parse the Excel Workbook

### 1.1 Install dependencies
```bash
pip install openpyxl --break-system-packages --quiet
```

### 1.2 Full-hardened parse script

```python
import openpyxl
import re
import json
from datetime import datetime, date

# ── Gherkin reserved words — must never become bare Scenario titles ────────────
GHERKIN_RESERVED = {
    'feature', 'background', 'scenario', 'scenario outline', 'examples',
    'given', 'when', 'then', 'and', 'but', 'rule', 'example',
}

def sanitize(name: str) -> str:
    """Normalise for comparison: lowercase, collapse whitespace, alphanum+underscore only."""
    return re.sub(r'[^a-z0-9_]', '', name.lower().strip().replace(' ', '_'))

def to_pascal(name: str) -> str:
    return ''.join(w.capitalize() for w in re.sub(r'[^a-z0-9 ]', '', name.lower()).split())

def to_snake(name: str) -> str:
    return re.sub(r'_+', '_', sanitize(name)).strip('_')

def cell_to_str(value) -> str:
    """Safely convert any openpyxl cell value type to a clean string."""
    if value is None:
        return ''
    if isinstance(value, bool):
        return str(value).lower()           # True→'true', False→'false'
    if isinstance(value, (datetime, date)):
        return value.strftime('%Y-%m-%d')   # Normalise all dates
    if isinstance(value, float):
        return str(int(value)) if value == int(value) else str(value)  # '1' not '1.0'
    return str(value).strip()

def unmerge_sheet(ws):
    """
    Expand all merged regions so every cell contains the anchor cell's value.
    Must be called before any row iteration.
    NOTE: openpyxl in read_only mode does not expose merged_cells; use normal mode.
    """
    for merge_range in list(ws.merged_cells.ranges):
        top_left_value = ws.cell(merge_range.min_row, merge_range.min_col).value
        ws.unmerge_cells(str(merge_range))
        for row in ws.iter_rows(
            min_row=merge_range.min_row, max_row=merge_range.max_row,
            min_col=merge_range.min_col, max_col=merge_range.max_col
        ):
            for cell in row:
                cell.value = top_left_value

def detect_header_row(ws) -> int:
    """
    Find the true header row (1-based).
    Handles workbooks with a title or blank row above the real header.
    Scans up to row 5; returns the first row that has >= 2 non-empty cells.
    Falls back to row 1 if nothing better is found.
    """
    for r in range(1, 6):
        non_empty = [c.value for c in ws[r] if c.value is not None and str(c.value).strip()]
        if len(non_empty) >= 2:
            return r
    return 1

def detect_column_roles(headers: list) -> dict:
    """
    Returns { header_name : role } where role is one of:
      'given' | 'when' | 'then' | 'and' | 'but' | 'tag' | 'comment' | 'skip' | 'data'
    """
    given_kw   = re.compile(r'^(given|pre[_\- ]?condition|setup|arrange|context)', re.I)
    when_kw    = re.compile(r'^(when|action|act|trigger|step|execute|do\b)', re.I)
    then_kw    = re.compile(r'^(then|expect|assert|verify|validate|check|result|outcome)', re.I)
    and_kw     = re.compile(r'^(and\b|also|additionally)', re.I)
    but_kw     = re.compile(r'^(but\b|except|however)', re.I)
    tag_kw     = re.compile(r'^(tag|label|category|suite|group)', re.I)
    comment_kw = re.compile(r'^(comment|note|description|remark)', re.I)

    roles = {}
    for h in headers:
        if h is None or not str(h).strip():
            roles[h] = 'skip'
        elif given_kw.match(str(h)):  roles[h] = 'given'
        elif when_kw.match(str(h)):   roles[h] = 'when'
        elif then_kw.match(str(h)):   roles[h] = 'then'
        elif and_kw.match(str(h)):    roles[h] = 'and'
        elif but_kw.match(str(h)):    roles[h] = 'but'
        elif tag_kw.match(str(h)):    roles[h] = 'tag'
        elif comment_kw.match(str(h)):roles[h] = 'comment'
        else:                         roles[h] = 'data'
    return roles

def parse_workbook(path: str) -> dict:
    # data_only=True returns computed values instead of formula strings
    wb = openpyxl.load_workbook(path, data_only=True)

    # Feature title priority: workbook title property → filename stem
    raw_title = getattr(wb.properties, 'title', None) or \
                path.split('/')[-1].rsplit('.', 1)[0]
    feature_title = raw_title.strip() or 'GeneratedFeature'

    warnings      = []
    skipped_tabs  = []
    scenarios     = {}
    background_tab = None
    tags_tab       = None

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]

        # ── Skip hidden sheets ────────────────────────────────────────────────
        if ws.sheet_state == 'hidden':
            skipped_tabs.append({'name': sheet_name, 'reason': 'hidden'})
            continue

        # ── Expand merged cells ───────────────────────────────────────────────
        unmerge_sheet(ws)

        sn_lower = sheet_name.strip().lower()

        # ── Detect special reserved tabs ──────────────────────────────────────
        if sn_lower == 'background':
            background_tab = sheet_name
            # Still parse normally; Phase 5.5 handles it as Background block
        if sn_lower in ('tag', 'tags', 'metadata'):
            tags_tab = sheet_name

        # ── Guard: bare Gherkin keyword as tab name ───────────────────────────
        if sn_lower in GHERKIN_RESERVED and sn_lower not in ('background',):
            warnings.append(
                f"Tab '{sheet_name}' is a Gherkin reserved keyword. "
                f"Rename it or Cucumber's parser may break."
            )
            if sn_lower in ('examples', 'rule', 'feature', 'scenario'):
                skipped_tabs.append({'name': sheet_name, 'reason': f'reserved Gherkin keyword: {sn_lower}'})
                continue

        # ── Detect header row ─────────────────────────────────────────────────
        header_row_idx = detect_header_row(ws)
        all_rows = list(ws.iter_rows(min_row=header_row_idx, values_only=True))

        if not all_rows:
            skipped_tabs.append({'name': sheet_name, 'reason': 'empty'})
            continue

        # ── Parse and deduplicate headers ─────────────────────────────────────
        raw_headers = all_rows[0]
        seen_hdrs   = {}
        headers     = []
        for i, h in enumerate(raw_headers):
            h_str = cell_to_str(h) if h is not None else f'_col{i}'
            if not h_str:
                h_str = f'_col{i}'
            if h_str in seen_hdrs:
                deduped = f'{h_str}_{i}'
                warnings.append(
                    f"Tab '{sheet_name}': duplicate header '{h_str}' renamed to '{deduped}'."
                )
                h_str = deduped
            seen_hdrs[h_str] = True
            headers.append(h_str)

        # ── Parse data rows ───────────────────────────────────────────────────
        data_rows = []
        seen_row_keys = []
        for row in all_rows[1:]:
            row_dict = {
                headers[i]: cell_to_str(v)
                for i, v in enumerate(row)
                if i < len(headers)
            }
            # Skip entirely empty rows
            if all(v == '' for v in row_dict.values()):
                continue
            # Detect multi-line cell content
            for k, v in row_dict.items():
                if '\n' in v:
                    warnings.append(
                        f"Tab '{sheet_name}', column '{k}': multi-line cell detected. "
                        f"Will use Gherkin DocString syntax."
                    )
            # Detect duplicate rows
            row_key = json.dumps(row_dict, sort_keys=True)
            if row_key in seen_row_keys:
                warnings.append(
                    f"Tab '{sheet_name}': duplicate data row detected → {row_dict}. "
                    f"Will produce duplicate Examples entries."
                )
            seen_row_keys.append(row_key)
            data_rows.append(row_dict)

        if not data_rows:
            skipped_tabs.append({'name': sheet_name, 'reason': 'headers only, no data'})
            warnings.append(f"Tab '{sheet_name}' has headers but no data rows — skipped.")
            continue

        column_roles = detect_column_roles(headers)

        # An outline needs 2+ rows OR any data-role columns (parameterised scenario)
        data_role_cols = [h for h, r in column_roles.items() if r == 'data']
        is_outline = len(data_rows) > 1 or (len(data_rows) == 1 and len(data_role_cols) > 0)

        # Warn if NO keyword-role columns exist at all (fully unroled tab)
        keyword_roles = [r for r in column_roles.values() if r in ('given','when','then','and','but')]
        if not keyword_roles:
            warnings.append(
                f"Tab '{sheet_name}': NO Given/When/Then column roles detected. "
                f"Will generate a skeleton with # TODO placeholders."
            )

        scenarios[sheet_name] = {
            'headers':       headers,
            'column_roles':  column_roles,
            'rows':          data_rows,
            'is_outline':    is_outline,
            'fully_unroled': not keyword_roles,
            'row_tags':      [],    # Populated from tags_tab if present
        }

    return {
        'feature_title':   feature_title,
        'scenarios':       scenarios,
        'background_tab':  background_tab,
        'tags_tab':        tags_tab,
        'skipped_tabs':    skipped_tabs,
        'warnings':        warnings,
    }

spec = parse_workbook("/mnt/user-data/uploads/<YOUR_FILE>.xlsx")
print(json.dumps(spec, indent=2, default=str))
```

### 1.3 Post-parse gate check

Log and review before continuing:
- Total tabs vs processable tabs vs skipped tabs
- All warnings
- Any `fully_unroled` tabs (require human review before generation)

**BLOCK conditions in Phase 1:**
- Zero processable tabs after all skips → stop, report to user
- Feature title empty after all fallbacks → use filename stem, warn user
- More than one tab has the exact same sanitized name → ask user which to keep

---

## Phase 2 — Discover Existing Project Artefacts

Run all scans from the **project root**. Substitute `TEST_ROOT` and `RESOURCE_ROOT`
from Phase 0.3 if they differ from `src/test/java` and `src/test/resources`.

### 2.1 Find all `.feature` files
```bash
find . \( -name "*.feature" \) \
     -not -path "*/target/*" \
     -not -path "*/build/*" \
     -not -path "*/.git/*" \
     -not -path "*/node_modules/*"
```

### 2.2 Extract Feature, Background, Scenario titles with line numbers
```bash
# Exclude commented-out lines (lines starting with optional whitespace then #)
grep -rn \
     -e "^\s*Feature:" \
     -e "^\s*Background:" \
     -e "^\s*Scenario:" \
     -e "^\s*Scenario Outline:" \
     -e "^\s*Rule:" \
     --include="*.feature" \
     --exclude-dir={target,build,.git} \
     . | grep -v "^\s*#"
```

### 2.3 Extract Background step sentences (for hoisting analysis in Phase 3.5)
```bash
awk '/^\s*Background:/,/^\s*(Scenario|Rule|@[A-Za-z])/' \
    $(find . -name "*.feature" -not -path "*/target/*" -not -path "*/build/*") 2>/dev/null \
    | grep -E "^\s+(Given|When|Then|And|But)"
```

### 2.4 Extract Step Definitions — Java annotation style
```bash
grep -rn \
     -e "@Given" -e "@When" -e "@Then" -e "@And" -e "@But" \
     --include="*.java" \
     --exclude-dir={target,build,.git} \
     . | grep -v "^Binary" | grep -v "import io.cucumber"
```

### 2.5 Extract Step Definitions — Kotlin annotation style
```bash
grep -rn \
     -e "@Given" -e "@When" -e "@Then" -e "@And" -e "@But" \
     --include="*.kt" \
     --exclude-dir={target,build,.git} \
     . | grep -v "import io.cucumber"
```

### 2.6 Extract Step Definitions — Java/Kotlin lambda style (En interface)
Lambda steps live in `init {}` blocks or constructors as method calls, not annotations:
```bash
grep -rn -E "^\s+(Given|When|Then|And|But)\s*\(" \
     --include="*.java" --include="*.kt" \
     --exclude-dir={target,build,.git} \
     .
```
> **Corner case:** If the lambda regex string spans multiple lines (rare but valid in
> Kotlin multi-line strings), flag it as MANUAL REVIEW — cannot reliably parse multi-line
> lambda step expressions with grep.

### 2.7 Extract `@ParameterType`, `@DataTableType`, `@DocStringType`
```bash
grep -rn \
     -e "@ParameterType" -e "@DataTableType" -e "@DocStringType" \
     --include="*.java" --include="*.kt" \
     --exclude-dir={target,build,.git} \
     .
```
Record these — new step expressions can use `{TypeName}` if a matching `@ParameterType`
already exists. Never invent a `{TypeName}` that has no registered transform.

### 2.8 Extract `@Before` / `@After` hooks
```bash
grep -rn \
     -e "@Before" -e "@After" -e "@BeforeAll" -e "@AfterAll" \
     --include="*.java" --include="*.kt" \
     --exclude-dir={target,build,.git} \
     . | grep -v "import" | grep -v "//\s*@"
```
Record each hook's class, tag filter expression, and `order` value.
New hooks must use a different `order` value to avoid silent ordering conflicts.

### 2.9 Detect the Cucumber runner class
```bash
grep -rn \
     -e "@RunWith(Cucumber" \
     -e "CucumberOptions" \
     -e "@ConfigurationParameter" \
     -e "SelectClasses.*Cucumber" \
     --include="*.java" --include="*.kt" \
     --exclude-dir={target,build,.git} \
     .
```
Record:
- `glue` paths (step definition packages)
- `features` paths (classpath resource paths)
- `tags` expression (new scenarios must carry a matching tag)
- `plugin` list

### 2.10 Detect Spring Boot test wiring
```bash
grep -rn \
     -e "@CucumberContextConfiguration" \
     -e "CucumberSpringConfiguration" \
     --include="*.java" --include="*.kt" \
     --exclude-dir={target,build,.git} \
     .
```
Record which class carries `@CucumberContextConfiguration`.
**BLOCK:** if two or more classes carry it → report conflict to user; Spring cannot boot.

### 2.11 Detect `cucumber.properties` and active profiles
```bash
find . -name "cucumber.properties" -not -path "*/target/*" -not -path "*/build/*"
find . \( -name "application-test.yml" -o -name "application-test.properties" \) \
     -not -path "*/target/*"
grep -rn "@ActiveProfiles" --include="*.java" --include="*.kt" . | grep -v target
```

### 2.12 Build the Artefact Catalogue
```
CATALOGUE = {
  "features":         { sanitize(feature_title) : file_path },

  "scenarios":        { sanitize(scenario_title) :
                          { "file": path, "line": int, "type": "Scenario|Outline" } },

  "backgrounds":      { sanitize(feature_title) :
                          [ list_of_given_step_sentences ] },

  "steps":            { raw_pattern :
                          { "file": class_path, "method": method_name,
                            "keyword": "Given|When|Then|And|But",
                            "style": "annotation|lambda" } },

  "hooks":            { "before": [ {class, tag_filter, order} ],
                        "after":  [ {class, tag_filter, order} ] },

  "runner":           { "glue": [...], "features": [...], "tags": "..." },

  "spring_config":    "<FQN of @CucumberContextConfiguration class or null>",

  "custom_types":     [ list of @ParameterType names ],
}
```

---

## Phase 3 — Collision and Conflict Analysis

Run every check below and produce a **Decision Matrix** for each tab before writing any file.

### 3.1 Scenario-level matching
```
For each (tab_name, tab_data) in spec["scenarios"]:
  key = sanitize(tab_name)

  A. Exact match in CATALOGUE["scenarios"]?
       → action = REUSE_SCENARIO
         Record (file_path, line_number).

  B. Near-duplicate (same sanitized base, differing trailing digits / suffixes)?
       Example: "login_1" vs existing "login"
       → action = WARN_NEAR_DUPLICATE
         Ask user: intentional new variant or accidental duplicate?

  C. No match → action = CREATE_SCENARIO
```

### 3.2 Feature-level matching
```
For each tab marked CREATE_SCENARIO:
  feature_key = sanitize(spec["feature_title"])

  A. CATALOGUE["features"] contains feature_key?
       → action = APPEND_TO_EXISTING_FEATURE (never overwrite)
  B. No match → action = CREATE_FEATURE_FILE
```

### 3.3 Step-level matching
For every step sentence Phase 4 will generate:

```python
def match_step(sentence: str, catalogue_steps: dict) -> dict:
    matches = []
    for pattern, meta in catalogue_steps.items():
        # First: Cucumber Expression check
        if cucumber_expression_matches(pattern, sentence):
            matches.append((pattern, meta))
        # Then: anchored regex check (strip ^ and $ for matching)
        else:
            clean_pattern = pattern.strip().lstrip('^').rstrip('$')
            if re.fullmatch(clean_pattern, sentence):
                matches.append((pattern, meta))

    if len(matches) == 0:
        return {"action": "CREATE_STEP"}
    if len(matches) == 1:
        return {"action": "REUSE_STEP", "pattern": matches[0][0], "meta": matches[0][1]}
    # AMBIGUOUS — multiple matches — NEVER silently pick one
    return {"action": "AMBIGUOUS_STEP", "matches": matches}
```

**If AMBIGUOUS_STEP:** BLOCK generation for that scenario.
Report all conflicting patterns to the user. Do not proceed until resolved.
Cucumber will throw `AmbiguousStepDefinitionsException` at runtime.

### 3.4 Step keyword mismatch check
Cucumber treats `@Given`, `@When`, `@Then` as interchangeable at runtime, but mixing
them degrades readability. If a matching existing step uses a different keyword than
what the Gherkin sentence requires:
```
⚠️ KEYWORD MISMATCH
   Step    : "user is logged in"
   Existing: @Given in LoginSteps.java:42
   Required: When (per column role in tab 'Purchase Flow')
   → Suggest: Use "And user is logged in" in the Gherkin, or add a @When alias.
```

### 3.5 Background hoisting analysis
```python
given_signatures = {}
for tab_name, tab_data in spec["scenarios"].items():
    given_hdrs = [h for h, r in tab_data["column_roles"].items() if r == "given"]
    given_vals = tuple(tab_data["rows"][0].get(h, "") for h in given_hdrs)
    given_signatures[tab_name] = (tuple(given_hdrs), given_vals)

all_sigs = list(given_signatures.values())
HOIST_TO_BACKGROUND = (
    len(all_sigs) > 1 and
    len(set(all_sigs)) == 1  # All tabs share identical Given headers AND values
)
```
If `HOIST_TO_BACKGROUND = True`:
- Given steps go into a `Background:` block
- Remove Given steps from every individual Scenario
- Check CATALOGUE["backgrounds"] — if target feature already has a Background block,
  compare it with the new one:
  - Identical → reuse existing, do nothing
  - Different → BLOCK: a Feature can have only one Background; report to user

### 3.6 Reserved keyword tab collision rules

| Tab name (lowercased)                | Rule                                                         |
|--------------------------------------|--------------------------------------------------------------|
| `background`                         | Parse as Background block definition (Phase 5.5)            |
| `examples`                           | BLOCK — rename required; breaks Cucumber parser             |
| `rule`, `feature`, `scenario`        | BLOCK — rename required; breaks Cucumber parser             |
| `given`, `when`, `then`, `and`, `but`| WARN — legal but confusing; append `_steps` to Scenario title |
| `tags`, `tag`, `metadata`            | Parse as tag configuration (Phase 5.6)                      |

### 3.7 Angle bracket `<placeholder>` collision check
For Scenario Outlines, `<ColumnName>` in step text are Cucumber placeholders.
Any `<text>` that is NOT an Examples column name will cause Cucumber to fail silently
or throw a parse error.

```python
def validate_placeholders(step_text: str, examples_headers: list, warnings: list):
    for match in re.finditer(r'<([^>]+)>', step_text):
        inner = match.group(1)
        if inner not in examples_headers:
            warnings.append(
                f"Step text contains '<{inner}>' which is NOT an Examples column. "
                f"Escape it or reword the step."
            )
```

### 3.8 Pipe character collision check
For Examples tables, any `|` inside a cell value must be escaped as `\|`.
Detect at parse time:
```python
for row in tab_data["rows"]:
    for k, v in row.items():
        if '|' in v:
            warnings.append(
                f"Tab '{tab_name}', column '{k}', value '{v}' contains '|'. "
                f"Will be escaped as '\\|' in the Examples table."
            )
```

---

## Phase 4 — Generate Gherkin from Excel Data

### 4.1 Step sentence construction — priority rules

**Priority 1: Header IS a full sentence (≥ 3 words, contains a verb)**
Use the header as the step template verbatim. Cell values become `<placeholder>` values.
```
Header: "User logs in with username"
Value : "alice"
Result: Given User logs in with <username>   (Outline)
Result: Given User logs in with "alice"      (plain Scenario)
```

**Priority 2: Header starts with a Gherkin keyword**
Strip the leading keyword to avoid doubling:
```
Header: "Given user is on the homepage"
Result: Given user is on the homepage     (NOT "Given Given user...")
```

**Priority 3: Short noun/phrase header (< 3 words or no verb)**
Construct: `<keyword> the <header> is "<value>"`
```
Header: "Status", Value: "Active"
Result: Given the status is "Active"
```

**Priority 4: data-role columns — no keyword affinity**
Attach to the closest preceding keyword step as an inline parameter.
If no keyword step exists above it, wrap all data columns in a DataTable step:
```gherkin
When the system receives the following data:
  | field    | value    |
  | colName1 | value1   |
  | colName2 | value2   |
```

### 4.2 Multi-line cell values → DocString
```gherkin
When the system processes the request:
  """
  First line of value
  Second line of value
  """
```
Step definition must accept `String` as the last parameter (or `DocString` in Cucumber 7+).

### 4.3 Structured data columns → DataTable
If 3+ `data`-role columns exist with no keyword role association:
```gherkin
When the user submits the form with:
  | field    | value     |
  | username | john_doe  |
  | password | secret123 |
```
Step definition:
```java
@When("the user submits the form with:")
public void theUserSubmitsFormWith(DataTable dataTable) {
    Map<String, String> data = dataTable.asMap(String.class, String.class);
}
```

### 4.4 Value type inference for Cucumber Expressions

| Cell value pattern               | Cucumber Expr        | Java type    | Notes                              |
|----------------------------------|----------------------|--------------|------------------------------------|
| Integer only (`^\d+$`)           | `{int}`              | `int`        | Use `{long}` if value > 2,147,483,647 |
| Decimal (`^\d+\.\d+$`)           | `{double}`           | `double`     | `{float}` if lower precision needed|
| ISO date `YYYY-MM-DD`            | `{string}` + parse   | `String`     | Parse with `LocalDate.parse()` in step |
| `true` or `false` (lowercase)    | `(true\|false)` regex | `boolean`   | No built-in Cucumber boolean type  |
| Single word, no spaces           | `{word}`             | `String`     | Fails if value ever has a space    |
| Multi-word / quoted string       | `{string}`           | `String`     | Wrap value in `"..."` in Gherkin   |
| Empty string                     | —                    | —            | Skip as parameter; warn user       |
| UUID (`xxxxxxxx-xxxx-...`)       | `{string}`           | `String`     |                                    |
| Enum-like (limited unique vals)  | `(val1\|val2\|val3)` | `String`     | Build from unique values across all rows |
| Custom `@ParameterType` in proj  | `{TypeName}`         | custom class | Only if found in Phase 2.7         |
| Large number (> 2^31)            | `{long}`             | `long`       |                                    |

> **Default rule:** When in doubt, use `{string}` and wrap the value in `"..."`.
> Never leave a multi-word parameter unquoted in the Gherkin.
> Never use `{string}` for numeric values that will be used in arithmetic.

### 4.5 Tag derivation (applied in this order)
1. Explicit `TAGS` tab (Phase 5.6) — highest priority
2. Column role = `tag` in the data tab → cell value becomes `@tagValue` on that scenario
3. Feature-level tag = `@<to_snake(feature_title)>` — always added to Feature block
4. Scenario-level tag = `@<to_snake(tab_name)>` — always added to each Scenario
5. `@automated` — always added to every generated Scenario
6. `@generated` — always added to every generated Scenario (marks AI-generated tests)

### 4.6 Scenario Outline — Examples table rules
- Column headers: only alphanumerics, spaces, hyphens, underscores. Strip everything else.
- Pipe characters `|` in cell values must be escaped as `\|`.
- Empty cells → substitute `N/A` and warn user (Cucumber cannot handle empty Examples cells).
- Duplicate rows → include with warning (Cucumber will run them; may be intentional).
- Max recommended columns: 8. If more, suggest splitting into multiple Outlines or DataTables.
- Row-level tags (Cucumber 6+): if a `tag` column exists, split Examples into one block per unique tag value.

### 4.7 Plain Scenario template (1 data row)
```gherkin
  # Generated from: <excel_filename>, tab: <tab_name>, date: <YYYY-MM-DD>
  @<scenario_snake_tag> @automated @generated
  Scenario: <Tab Name — preserved original casing and spaces>
    Given <given_step_sentence_1>
    And   <and_step_sentence if present>
    When  <when_step_sentence_1>
    And   <and_step_sentence if present>
    Then  <then_step_sentence_1>
    And   <then_continuation if present>
    But   <but_step_sentence if present>
```

### 4.8 Scenario Outline template (2+ data rows)
```gherkin
  # Generated from: <excel_filename>, tab: <tab_name>, date: <YYYY-MM-DD>
  @<scenario_snake_tag> @automated @generated
  Scenario Outline: <Tab Name — preserved original casing and spaces>
    Given <given template using <ColName1> and <ColName2>>
    When  <when template using <ColName3>>
    Then  <then template using <ColName4>>

    @<row_tag_if_present>
    Examples:
      | ColName1  | ColName2  | ColName3  | ColName4  |
      | val_r1_c1 | val_r1_c2 | val_r1_c3 | val_r1_c4 |
      | val_r2_c1 | val_r2_c2 | val_r2_c3 | val_r2_c4 |
```

### 4.9 Background block template (if HOIST_TO_BACKGROUND = True)
```gherkin
  Background:
    # Shared preconditions extracted from all scenario tabs
    Given <shared_given_step_1>
    And   <shared_given_step_2>
```
Placed immediately after the Feature narrative and before the first Scenario.

### 4.10 Fully unroled tabs — skeleton template
If ALL columns have role = `data` (no Given/When/Then headers detected):
```gherkin
  # ⚠️ MANUAL REVIEW REQUIRED
  # Tab '<TabName>' has no Given/When/Then column roles detected.
  # Headers found: [col1, col2, col3, ...]
  # Assign keyword roles by prefixing headers with 'Given ', 'When ', 'Then '
  # then regenerate.
  @<scenario_snake_tag> @automated @generated @needs_review
  Scenario: <Tab Name>
    # TODO: Given ...
    # TODO: When  ...
    # TODO: Then  ...
```

### 4.11 Tab names that are pure numbers or start with a digit
```
Tab name: "001" or "42 Login"
Gherkin Scenario title: "Scenario 001" or "42 Login"  ← prefix with "Scenario " if pure number
Java class name: Must not start with digit → prefix with "S_" → "S_001Steps"
```

---

## Phase 5 — Write Artefacts (Reuse-First, Create-Second)

### 5.1 Golden rules — NEVER violate these
1. **Never overwrite** an existing `.feature` file — always append.
2. **Never delete** any existing Scenario, even if it appears superseded.
3. **Never duplicate** a step definition regex/expression — check CATALOGUE before every write.
4. **Never add** `@SpringBootTest` or `@CucumberContextConfiguration` to new Step Definition classes.
5. **Never place** new `.feature` files outside the detected `RESOURCE_ROOT`.
6. **Always prepend** a generation comment above new Scenarios.
7. **Always match** the line-ending style (LF vs CRLF) of the target file.
8. **Always use** UTF-8 encoding without BOM for all written files.

### 5.2 Line ending and encoding safety
```python
def read_file_safe(path: str) -> tuple[str, str]:
    """Returns (content, line_ending) where line_ending is '\r\n' or '\n'."""
    with open(path, 'rb') as f:
        raw = f.read()
    # Strip BOM if present
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    content = raw.decode('utf-8')
    line_ending = '\r\n' if '\r\n' in content else '\n'
    return content, line_ending

def write_file_safe(path: str, content: str, line_ending: str = '\n'):
    """Writes UTF-8 without BOM, preserving specified line endings."""
    normalized = content.replace('\r\n', '\n').replace('\r', '\n')
    if line_ending == '\r\n':
        normalized = normalized.replace('\n', '\r\n')
    with open(path, 'wb') as f:
        f.write(normalized.encode('utf-8'))
```

### 5.3 Feature file write strategy
```
feature_key = sanitize(spec["feature_title"])

IF feature_key in CATALOGUE["features"]:
    target_file = CATALOGUE["features"][feature_key]
    action = APPEND
    # Read file first; verify no Scenario with same title already at bottom
ELSE:
    domain    = infer_domain(CATALOGUE)   # See helper below
    filename  = to_snake(spec["feature_title"]) + ".feature"
    target_path = f"{RESOURCE_ROOT}/features/{domain}/{filename}"
    action = CREATE
    # Create directory if it doesn't exist:
    # os.makedirs(os.path.dirname(target_path), exist_ok=True)
    # Write Feature header first (Phase 6.1 template)
```

#### Domain inference helper
```python
from collections import Counter
import os

def infer_domain(catalogue: dict) -> str:
    """
    Inspect directory names of existing feature files.
    Match new feature title keywords against existing domain directories.
    Fall back to "" (flat structure) if no match found.
    """
    dirs = []
    for path in catalogue["features"].values():
        parts = path.replace('\\', '/').split('/')
        if 'features' in parts:
            idx = parts.index('features')
            if idx + 1 < len(parts) - 1:   # At least one subdirectory below features/
                dirs.append(parts[idx + 1])
    if not dirs:
        return ""
    most_common = Counter(dirs).most_common(1)[0][0]
    return most_common
```

### 5.4 Append to existing feature file
```python
def append_to_feature(file_path: str, new_gherkin: str):
    content, line_ending = read_file_safe(file_path)
    sep = line_ending * 2
    if not content.endswith(sep):
        content = content.rstrip(line_ending) + sep
    write_file_safe(file_path, content + new_gherkin, line_ending)
```

### 5.5 Step Definition write strategy
```
For each new step marked CREATE_STEP:

  1. Find the most relevant EXISTING Step Definition class:
     a. File whose name starts with same feature/domain keyword (case-insensitive)
     b. Class in the same package as steps already matched via REUSE_STEP
     c. No match → CREATE new class (template in Phase 6.3 / 6.4)

  2. Append new method to chosen existing class:
     a. Read the file
     b. Locate the last closing brace `}` of the class body
     c. Insert new method BEFORE that closing brace
     d. Add any missing imports at the top of the file (after package declaration)
     e. NEVER add a method whose expression/regex already exists in this file

  3. Method uniqueness check within the class (not just project-wide):
     grep -c "@Given(\"<expression>\")" <target_class_file>
     → If count > 0 → skip, already present in this class
```

### 5.6 Handling the explicit BACKGROUND tab
If `spec["background_tab"]` is set:
- Parse it as a normal tab to extract Given/When/Then steps
- Generate a `Background:` block from those steps
- Check if the target `.feature` file already has a `Background:` block
  - YES and IDENTICAL → reuse existing, do nothing
  - YES and DIFFERENT → BLOCK: one Background per Feature; report to user
  - NO → insert Background block after Feature narrative, before first Scenario

### 5.7 Handling the TAGS tab
If `spec["tags_tab"]` is set, parse it with expected columns:
```
| scope          | target              | tag        |
| feature        | <feature_title>     | smoke      |
| scenario       | <tab_name>          | regression |
| examples_row   | <tab_name>:<row_idx>| edge_case  |
```
Apply each tag during Gherkin generation in Phase 4.5.

### 5.8 Spring Boot context — CucumberSpringConfiguration check
After writing all Step Definition files:
```bash
grep -rn "CucumberContextConfiguration" \
     --include="*.java" --include="*.kt" \
     --exclude-dir={target,build} .
```

**If NONE found**, create:
```java
package <BASE_PACKAGE>;

import io.cucumber.spring.CucumberContextConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

/**
 * Central Spring context bootstrap for all Cucumber step definitions.
 * DO NOT add @SpringBootTest to individual step definition classes.
 */
@CucumberContextConfiguration
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
public class CucumberSpringConfiguration {
    // Intentionally empty — Spring context is bootstrapped here for all steps.
}
```

> **Cucumber 6 note:** Import is `io.cucumber.spring.CucumberContextConfiguration`
> **Cucumber 7 note:** Same import, but the runner uses `@Suite` + JUnit Platform

**If TWO OR MORE found** → BLOCK. Report conflict to user. Spring cannot create
two application contexts for the same Cucumber run.

### 5.9 ScenarioContext / Shared State
Check for an existing context holder:
```bash
grep -rn "ScenarioContext\|WorldObject\|TestContext\|StepContext\|ScenarioScope" \
     --include="*.java" --include="*.kt" . | grep -v target
```

If new steps need to pass data between step methods and no holder exists:
```java
package <BASE_PACKAGE>.context;

import io.cucumber.java.Scenario;
import org.springframework.stereotype.Component;
import org.springframework.web.context.annotation.ScenarioScope;
import java.util.HashMap;
import java.util.Map;

/**
 * Scenario-scoped bean for sharing state between step definitions.
 * @ScenarioScope ensures a fresh instance per Cucumber scenario.
 * Inject with @Autowired in any Step Definition class.
 */
@Component
@ScenarioScope
public class ScenarioContext {
    private final Map<String, Object> store = new HashMap<>();
    private Scenario currentScenario;

    public void setScenario(Scenario scenario) { this.currentScenario = scenario; }
    public Scenario getScenario()              { return currentScenario; }

    public void set(String key, Object value)  { store.put(key, value); }

    @SuppressWarnings("unchecked")
    public <T> T get(String key, Class<T> type) {
        Object val = store.get(key);
        if (val == null) throw new IllegalStateException(
            "ScenarioContext key '" + key + "' not found. " +
            "Ensure the step that sets it runs before the step that reads it."
        );
        return type.cast(val);
    }

    public boolean contains(String key) { return store.containsKey(key); }
    public void clear()                  { store.clear(); }
}
```

> **Critical:** `@ScenarioScope` is mandatory. Without it, Spring defaults to singleton
> scope, causing state to leak across scenarios and producing non-deterministic failures.

### 5.10 @Before / @After hooks for new features
Only create hooks if setup/teardown columns were detected in the Excel data
(column names containing "setup", "teardown", "cleanup", "reset", "precondition").
Check existing hook classes first (Phase 2.8) to avoid order conflicts.

```java
package <BASE_PACKAGE>.hooks;

import io.cucumber.java.Before;
import io.cucumber.java.After;
import io.cucumber.java.Scenario;
import org.springframework.beans.factory.annotation.Autowired;
// import <BASE_PACKAGE>.context.ScenarioContext;

/**
 * Lifecycle hooks for @<feature_snake_tag> scenarios.
 * order = 200 (adjust if conflicts with existing hooks at order 100).
 */
public class <FeatureTitlePascalCase>Hooks {

    // @Autowired
    // private ScenarioContext scenarioContext;

    @Before(value = "@<feature_snake_tag>", order = 200)
    public void setUp(Scenario scenario) {
        // scenarioContext.setScenario(scenario);
        // TODO: feature-specific setup
    }

    @After(value = "@<feature_snake_tag>", order = 200)
    public void tearDown(Scenario scenario) {
        if (scenario.isFailed()) {
            // TODO: attach screenshot or log on failure
            // scenario.attach(screenshot, "image/png", "failure-screenshot");
        }
        // scenarioContext.clear();
        // TODO: feature-specific teardown
    }
}
```

### 5.11 Runner class — glue and features path update
After writing all artefacts, verify the runner class covers new packages and paths.

**Cucumber 6 (@CucumberOptions):**
```java
@CucumberOptions(
    features = { "src/test/resources/features" },    // covers all subdirs
    glue     = {
        "<BASE_PACKAGE>",           // CucumberSpringConfiguration lives here
        "<BASE_PACKAGE>.steps",     // all step definitions
        "<BASE_PACKAGE>.hooks"      // all hooks
    },
    plugin   = { "pretty", "html:target/cucumber-reports/index.html" },
    tags     = "@automated"         // matches @automated added to all generated scenarios
)
```

**Cucumber 7 (@ConfigurationParameter):**
```java
@ConfigurationParameter(key = GLUE_PROPERTY_NAME,
    value = "<BASE_PACKAGE>,<BASE_PACKAGE>.steps,<BASE_PACKAGE>.hooks")
@ConfigurationParameter(key = FEATURES_PROPERTY_NAME,
    value = "src/test/resources/features")
@ConfigurationParameter(key = FILTER_TAGS_PROPERTY_NAME,
    value = "@automated")
```

If the runner's current `glue` or `features` values are narrower than needed,
update them. If they use classpath notation (`classpath:features/auth`), widen to
`classpath:features` to cover the new domain directory.

---

## Phase 6 — Full File Templates

### 6.1 Complete new `.feature` file
```gherkin
# =============================================================================
# Feature  : <Feature Title>
# Source   : <excel_filename>
# Generated: <YYYY-MM-DD>
# =============================================================================

@<feature_snake_tag> @automated @generated
Feature: <Feature Title>
  As a <persona — infer from context or use "user">
  I want to <action — derived from When column headers>
  So that <goal — derived from Then column headers>

  # ---------------------------------------------------------------------------
  # Background — shared preconditions (hoisted from all scenario tabs)
  # ---------------------------------------------------------------------------
  Background:                         ← include only if HOIST_TO_BACKGROUND = True
    Given <shared_given_step_1>
    And   <shared_given_step_2>

  # ---------------------------------------------------------------------------
  # Generated from tab: <TabName>
  # ---------------------------------------------------------------------------

  @<scenario_snake_tag> @automated @generated
  Scenario: <Tab Name>
    Given <step>
    When  <step>
    Then  <step>
```

### 6.2 Scenario Outline in full context
```gherkin
  # Generated from tab: <TabName> — <N> data rows → Scenario Outline
  @<scenario_snake_tag> @automated @generated
  Scenario Outline: <Tab Name>
    Given <template with <ColName1>>
    When  <template with <ColName2>>
    Then  <template with <ColName3>>

    Examples:
      | ColName1  | ColName2  | ColName3  |
      | row1_val1 | row1_val2 | row1_val3 |
      | row2_val1 | row2_val2 | row2_val3 |
```

### 6.3 New Java Step Definition class (full)
```java
package <BASE_PACKAGE>.steps;

import io.cucumber.java.en.Given;
import io.cucumber.java.en.When;
import io.cucumber.java.en.Then;
import io.cucumber.datatable.DataTable;
import org.springframework.beans.factory.annotation.Autowired;
import static org.assertj.core.api.Assertions.assertThat;
import java.util.List;
import java.util.Map;
// import <BASE_PACKAGE>.context.ScenarioContext;

/**
 * Step definitions for Feature: <FeatureTitle>
 * Generated from: <excel_filename> on <YYYY-MM-DD>
 *
 * ⚠️ Do NOT add @SpringBootTest or @CucumberContextConfiguration here.
 *    Spring wiring is provided by: <spring_config_class>
 */
public class <FeatureTitlePascalCase>Steps {

    // Uncomment and add services as needed — Spring injects them via context
    // @Autowired
    // private YourService yourService;

    // Uncomment to share state between steps in this scenario
    // @Autowired
    // private ScenarioContext scenarioContext;

    @Given("<cucumber expression or anchored regex>")
    public void <methodName>(<params>) {
        // TODO: implement
    }

    @When("<cucumber expression or anchored regex>")
    public void <methodName>(<params>) {
        // TODO: implement
    }

    @Then("<cucumber expression or anchored regex>")
    public void <methodName>(<params>) {
        // TODO: implement — use assertThat() for assertions
    }

    // DataTable example — add only if DataTable steps are needed
    @When("<step text>:")
    public void <methodName>(DataTable dataTable) {
        List<Map<String, String>> rows = dataTable.asMaps(String.class, String.class);
        rows.forEach(row -> {
            // TODO: process each row
        });
    }

    // DocString example — add only if multi-line cell steps are needed
    @When("<step text>:")
    public void <methodName>(String docString) {
        // TODO: process docString content
    }
}
```

### 6.4 New Kotlin Step Definition class (annotation style — preferred in Cucumber 7+)
```kotlin
package <BASE_PACKAGE>.steps

import io.cucumber.java.en.Given
import io.cucumber.java.en.When
import io.cucumber.java.en.Then
import io.cucumber.datatable.DataTable
import org.springframework.beans.factory.annotation.Autowired
import org.assertj.core.api.Assertions.assertThat
// import <BASE_PACKAGE>.context.ScenarioContext

/**
 * Step definitions for Feature: <FeatureTitle>
 * Generated from: <excel_filename> on <YYYY-MM-DD>
 *
 * ⚠️ Do NOT add @SpringBootTest or @CucumberContextConfiguration here.
 */
class <FeatureTitlePascalCase>Steps {

    // @Autowired
    // lateinit var yourService: YourService

    // @Autowired
    // lateinit var scenarioContext: ScenarioContext

    @Given("<cucumber expression>")
    fun `<human readable description>`(<params>) {
        // TODO: implement
    }

    @When("<cucumber expression>")
    fun `<human readable description>`(<params>) {
        // TODO: implement
    }

    @Then("<cucumber expression>")
    fun `<human readable description>`(<params>) {
        // TODO: implement — use assertThat() for assertions
    }
}
```

---

## Phase 7 — Cucumber Expression and Regex Rules

### 7.1 Prefer Cucumber Expressions over raw regex
```
✅ PREFERRED:  @Given("the user {string} has {int} items in the {word} cart")
❌ AVOID:      @Given("^the user \"(.+)\" has (\\d+) items in the (\\w+) cart$")
```
Use raw regex only when the pattern requires optional groups, lookaheads, or alternation
that Cucumber Expressions cannot express.

### 7.2 Always anchor raw regex
```java
@Given("^the user is on the (.+) page$")   // ✅ anchored — exact match only
@Given("the user is on the (.+) page")     // ❌ unanchored — risk of partial match
```

### 7.3 Full Cucumber Expression type table

| Data value pattern            | Cucumber Expr           | Java / Kotlin type | Notes                              |
|-------------------------------|-------------------------|--------------------|------------------------------------|
| Whole number                  | `{int}`                 | `int` / `Int`      | Use `{long}` / `Long` for big nums |
| Decimal                       | `{double}`              | `double` / `Double`|                                    |
| Quoted / multi-word string    | `{string}`              | `String`           | Strips surrounding quotes          |
| Single word                   | `{word}`                | `String`           | Fails if value has spaces          |
| ISO date                      | `{string}` + parse      | `String`           | Parse with `LocalDate.parse()`     |
| Boolean `true`/`false`        | `(true\|false)` regex   | `boolean`/`Boolean`| No built-in Cucumber bool type     |
| Enum-like fixed set           | `(optA\|optB\|optC)` regex | `String`        | Derive from unique Excel values    |
| UUID                          | `{string}`              | `String`           |                                    |
| Custom project type           | `{TypeName}`            | custom class       | Only if `@ParameterType` exists    |
| Unconstrained (last resort)   | `{}`                    | custom `@ParameterType` | Must have matching transform  |

### 7.4 Naming conventions for step methods

| Step text                           | Java method                         | Kotlin function                    |
|-------------------------------------|-------------------------------------|------------------------------------|
| `Given the user is logged in`       | `theUserIsLoggedIn()`               | `` `the user is logged in`() ``   |
| `When the user submits {string}`    | `theUserSubmits(String formName)`   | `` `the user submits`(formName: String) `` |
| `Then the response status is {int}` | `theResponseStatusIs(int code)`     | `` `the response status is`(code: Int) `` |

Method names must be unique within the class.
If two steps have very similar text, disambiguate with a suffix:
`theUserIsLoggedInAsAdmin()` vs `theUserIsLoggedInAsGuest()`.

### 7.5 Step expression escaping rules
These characters have special meaning in Cucumber Expressions and must be escaped
or avoided in step text when not intended as type parameters:
- `{` and `}` → only use for typed parameters like `{int}`, `{string}`
- `(` and `)` → only use for optional text or regex alternation
- `/` → not special but avoid in expressions to prevent confusion
- `\` → escape character in regex; double-escape in Java string literals

---

## Phase 8 — Hardened Validation Checklist

Run every item. Mark ✅ PASS or ❌ FAIL with detail.
**Do not present any output to the user until all items are ✅ PASS or have a documented SKIP justification.**

### 8.1 Excel Parse Validation
```
[ ] All workbook tabs accounted for (processed, skipped, or blocked — none silently dropped)
[ ] No hidden-tab data included in output
[ ] Merged cells fully expanded — zero stray None/null values in any data row
[ ] Datetime cells converted to ISO YYYY-MM-DD string (not Python datetime objects)
[ ] Float cells representing whole numbers rendered without decimal ('1' not '1.0')
[ ] Boolean cells rendered as lowercase 'true'/'false'
[ ] Duplicate header names in any tab → deduplicated and warned
[ ] Duplicate data rows in any tab → included with warning
[ ] Multi-line cell content → flagged for DocString; warning issued
[ ] Tabs with only headers and no data → skipped with warning; not silently omitted
[ ] Tabs with Gherkin reserved names → handled per Phase 3.6 rules; not silently generated
[ ] BOM stripped from Excel file if present
[ ] Feature title is non-empty after all fallbacks
```

### 8.2 Gherkin Correctness
```
[ ] Every processed tab has a corresponding Scenario or Scenario Outline in output
[ ] Feature: block appears exactly once per .feature file
[ ] Background: block appears at most once per .feature file
[ ] No Scenario title is empty or whitespace-only
[ ] No duplicate Scenario titles within the same .feature file
[ ] Every <placeholder> in Scenario Outline steps has a matching column in Examples:
[ ] No <placeholder> syntax appears in plain Scenario steps (only Outlines use < >)
[ ] Pipe characters | in Examples values are escaped as \|
[ ] Examples tables have at least one data row
[ ] DocString blocks use exactly three double-quotes """ and are properly indented
[ ] DataTable blocks have consistent column counts in every row (including header)
[ ] All tags start with @ and contain only alphanumerics, hyphens, or underscores
[ ] No step text is empty or whitespace-only
[ ] Feature file is valid UTF-8; BOM absent; consistent LF or CRLF line endings
[ ] Generation comment block present above each generated Scenario
[ ] @automated and @generated tags present on every new Scenario
```

### 8.3 Step Definition Correctness
```
[ ] No two step definitions in the entire project share the same expression/regex (ambiguity)
[ ] Every new method has a unique name within its class
[ ] All required imports present (io.cucumber.java.en.*, DataTable, assertThat, etc.)
[ ] No new Step Def class carries @SpringBootTest or @CucumberContextConfiguration
[ ] @Autowired fields declared for services that are actually injected
[ ] ScenarioContext injected only where inter-step state is needed; not added everywhere
[ ] Method parameter count matches capture groups / typed params in the expression
[ ] All step methods have void return type
[ ] DocString step methods accept String or DocString as their last (and only extra) param
[ ] DataTable step methods accept DataTable or List<Map<String,String>> as their last param
[ ] Kotlin backtick method names contain no characters that break Kotlin compilation
[ ] All new step packages are within the runner's glue path
[ ] No raw regex expression is missing ^ and $ anchors
```

### 8.4 Spring Boot Wiring
```
[ ] Exactly one class in the project carries @CucumberContextConfiguration
[ ] That class carries @SpringBootTest (or a valid test slice annotation)
[ ] @ActiveProfiles("test") present on the config class (or in cucumber.properties)
[ ] All new step packages are covered by the runner's glue configuration
[ ] All new feature directories are covered by the runner's features configuration
[ ] @ScenarioScope on ScenarioContext bean (not singleton, not request scope)
[ ] New hooks use a unique @Order value not already taken by existing hooks
[ ] No circular @Autowired dependencies introduced by new step classes
[ ] Hooks use tag filters (@Before(value="@tag")) to avoid running for unrelated scenarios
```

### 8.5 File Write Safety
```
[ ] No existing .feature file was overwritten (check via diff before/after)
[ ] No existing Step Definition class was overwritten
[ ] No existing step method was deleted or modified
[ ] New Scenarios appended to existing feature files are separated by exactly one blank line
[ ] New methods appended to existing step classes are inside the class body (before final })
[ ] No new files were created outside src/test/ subtree (or detected TEST_ROOT / RESOURCE_ROOT)
[ ] All new files use UTF-8 encoding without BOM
[ ] Directory structure created (os.makedirs with exist_ok=True) before writing new files
```

### 8.6 Runner coverage check
```bash
# Verify glue covers all new step packages
grep -n "glue\|GLUE" \
     $(find . -name "*Runner*" -o -name "*CucumberTest*" | grep -v target) \
     2>/dev/null

# Verify features path covers new feature directories
grep -n "features\|FEATURES" \
     $(find . -name "*Runner*" -o -name "*CucumberTest*" | grep -v target) \
     2>/dev/null
```

### 8.7 Compilation pre-check (strongly recommended)
```bash
# Maven
mvn test-compile -q 2>&1 | tail -30

# Gradle
./gradlew testClasses 2>&1 | tail -30
```
If compilation fails: report the exact error, identify the generated file responsible,
and fix before presenting output to the user.

---

## Phase 9 — Output Summary to User

Present this full report after all files are written and the checklist is green:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AUTOMATION GENERATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📥 INPUT
   File         : <filename>.xlsx
   Format       : xlsx | xls (converted)
   Total tabs   : <N>
   Processed    : <N>
   Skipped      : <N>
     <tab_name> — reason: hidden | empty | reserved keyword | headers only

📋 FEATURE MAPPING
   Feature title : <title>
   Feature file  : <path>  [NEW | APPENDED]
   Background    : Hoisted from all tabs | Explicit BACKGROUND tab | None

♻️  REUSED ARTEFACTS
   Scenarios  : <list: "Scenario Title" → existing_file.feature:line>
   Step Defs  : <list: @Keyword("expression") → ClassName.methodName()>

✨ CREATED ARTEFACTS
   Feature files      : <list of new .feature paths>
   Appended features  : <list of existing .feature paths that were appended to>
   New scenarios      : <list of new Scenario / Scenario Outline titles>
   Step Def files     : <list of new Step Definition class paths>
   Modified Step Defs : <list of existing classes with new methods appended>
   New step methods   : <list of @Keyword("expression") → ClassName.methodName()>
   Spring config      : CucumberSpringConfiguration created | already existed | not needed
   ScenarioContext    : Created | already existed | not needed
   Hooks              : <FeatureName>Hooks created | existing hooks reused | none needed

⚠️  ACTION REQUIRED — Must be resolved before running tests
   AMBIGUOUS STEPS    : <step sentence → [conflicting_pattern_1, conflicting_pattern_2]>
   KEYWORD MISMATCH   : <sentence → existing @Given used as When; suggestion given>
   NEAR DUPLICATES    : <new tab "login_1" resembles existing "login" — intentional?>
   UNROLED TABS       : <tab names with no Given/When/Then headers → manual review>
   COMPILATION STATUS : PASSED | FAILED → <error snippet + file:line>
   RUNNER UPDATE      : <what was changed in runner, or "no changes needed">

📝  TODO LIST — Step methods needing implementation
   <ClassName>.java : <methodName>() — expression: "<step text>"
   ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Appendix A — Comprehensive Corner Case Reference

| Corner Case | Detection Method | Resolution |
|---|---|---|
| `.xls` (legacy format) | `file` command shows non-2007 Excel | Convert with xlrd (Phase 0.1a) |
| Password-protected workbook | `BadZipFile` on `load_workbook` | BLOCK → ask user to remove password |
| HTML disguised as `.xls` | `file` returns HTML | BLOCK → ask for real `.xlsx` |
| Hidden sheets | `ws.sheet_state == 'hidden'` | Skip silently; include in skipped_tabs |
| Merged cells | `ws.merged_cells.ranges` non-empty | Unmerge before any row iteration (Phase 1.2) |
| Datetime cells | `isinstance(value, datetime)` True | Format to `YYYY-MM-DD` |
| Date-only cells | `isinstance(value, date)` True | Format to `YYYY-MM-DD` |
| Float-as-integer cells | `1.0`, `42.0` | Cast to `int` before stringify |
| Boolean cells | Python `True`/`False` | Stringify as lowercase `'true'`/`'false'` |
| Formula cells (data_only=False) | `=SUM(...)` appears as string | Always use `data_only=True` in load_workbook |
| Empty formula result | `None` from `data_only=True` | Treat as empty string; warn user formula had no cached value |
| Duplicate header names | Same string in two positions in row 1 | Append column index to deduplicate; warn |
| Header row is not row 1 | First row has < 2 non-empty cells | Scan to row 5 for real header row |
| Multi-row headers | Row 1 is section title, row 2 is headers | `detect_header_row` picks row 2 automatically |
| Empty data rows | All cells None/empty | Skip; never include in Examples |
| Multi-line cell content | `\n` in cell string | Flag for DocString; warn user |
| Pipe in cell value | `\|` needed in Examples | Escape as `\|` in Examples table |
| `<text>` in step (non-placeholder) | Angle brackets not in Examples headers | Escape or reword; phase 3.7 check |
| Tab name is pure number | `"001"`, `"42"` | Prefix Scenario title with "Scenario "; prefix Java class with "S_" |
| Tab name starts with digit | `"1_Login"` | Prefix Java class name with `S_` or `C_` |
| Tab name has special file chars | `/`, `\`, `:`, `*`, `?`, `[`, `]` | Strip for filename; preserve in Scenario title |
| Tab name is leading/trailing whitespace | `"  Login  "` | `.strip()` before all uses |
| Tab name is a Gherkin keyword | `background`, `examples`, `rule`, `feature`, `scenario` | Handle or block per Phase 3.6 |
| Tab name has `given`/`when`/`then` | Legal but confusing | Append `_steps` to Scenario title; warn |
| All tabs share identical Given values | `HOIST_TO_BACKGROUND = True` | Extract Background block (Phase 3.5) |
| Background block already in feature file | Grep finds existing `Background:` | Check identity; block if different |
| Two Background tabs | Second `BACKGROUND` tab encountered | BLOCK: one Background per feature |
| Ambiguous step (2+ matches) | `match_step` returns `AMBIGUOUS_STEP` | BLOCK generation for that scenario |
| Step keyword mismatch | `@Given` exists but step needs `@When` | Warn; suggest `And` prefix or alias |
| Lambda-style step definitions | `Given(...)` in init/constructor, not annotated | Grep with `^\s+(Given\|When\|Then)\s*\(` |
| Step in abstract base class | Not a concrete class; harder to grep | Search for `abstract class` + `@Given` etc. |
| Step in external library/JAR | Not in source; grep returns nothing | Cannot discover; note in output as "may exist in dependency" |
| `@ParameterType` used but not found | `{TypeName}` in template for undeclared type | Fall back to `{string}`; warn |
| Missing `@CucumberContextConfiguration` | Grep finds zero matches | Create CucumberSpringConfiguration (Phase 5.8) |
| Two `@CucumberContextConfiguration` classes | Grep finds 2+ matches | BLOCK → report conflict |
| `@SpringBootTest` on step def class | Wrong class carries the annotation | Warn; it will cause duplicate context load |
| `@ScenarioScope` missing on context bean | Singleton context leaks between scenarios | Add `@ScenarioScope` (Phase 5.9) |
| Hook `@Order` conflict | New hook same order as existing | Use a different order value |
| Runner glue too narrow | New step package not in glue array | Widen glue to base package or add new package |
| Runner features path too narrow | New feature dir not under configured path | Widen to `src/test/resources/features` |
| `classpath:features/auth` path | Only scans one domain dir | Widen to `classpath:features` |
| Multi-module Maven project | Multiple `pom.xml` found | Identify test module; scope all writes to it |
| Non-standard test source root | `src/it/java`, `src/itest` | Use discovered `TEST_ROOT`; never assume default |
| Kotlin + Java mixed project | Both `.java` and `.kt` in test scope | Match language of nearest existing step def class |
| CRLF line endings in existing file | `\r\n` in file content | Detect and preserve; do not convert |
| BOM in UTF-8 file | `\xef\xbb\xbf` at file start | Strip on read; never write BOM |
| Feature file encoding not UTF-8 | Bytes outside ASCII range | Attempt UTF-8; fall back to Latin-1; warn |
| Commented-out scenarios in feature | `# Scenario:` in file | Exclude from CATALOGUE; do not count as existing |
| Scenario title mismatch due to whitespace | `"Login Test"` vs `"Login  Test"` | Always normalise with `sanitize()` before compare |
| Near-duplicate scenario names | `"login"` vs `"login_1"` | Warn; ask user if intentional new variant |
| Duplicate scenario titles across features | Two features both have `"Scenario: Login"` | Legal (different Features); warn for clarity |
| No clearly-roled columns | All columns have role `data` | Generate skeleton with `# TODO:` steps; tag `@needs_review` |
| Examples table with > 8 columns | Wide table, hard to read | Suggest splitting into multiple Outlines or DataTable |
| Row-level tags (Cucumber 6+) | `tag` column in data tab | Split Examples into one block per unique tag value |
| DataTable step with mismatched row width | Row has fewer/more pipes than header | Validate column count before writing |
| DocString indentation | Gherkin requires DocString indented 6 spaces beyond step | Apply correct indentation in template |
| Scenario Outline with single data row | `is_outline=True` due to data columns | Still generate as Outline; do not special-case to Scenario |
| `cucumber.properties` missing | No `glue` or `features` configured there | Use runner class annotations as source of truth |
| `@Tag` filter on runner excludes new scenarios | Runner has `tags="@smoke"` but new scenarios have different tag | Add matching tag to new scenarios, or update runner tags |
| No `assertThat` import | New step class missing AssertJ import | Always include `import static org.assertj.core.api.Assertions.assertThat` |

---

## Appendix B — Reference Project Layout

### Standard Maven / Gradle layout
```
src/
└── test/
    ├── java/
    │   └── com/example/
    │       ├── CucumberSpringConfiguration.java   ← @CucumberContextConfiguration + @SpringBootTest
    │       ├── CucumberRunnerTest.java             ← @CucumberOptions / @Suite
    │       ├── steps/
    │       │   ├── LoginSteps.java
    │       │   └── PaymentSteps.java
    │       ├── hooks/
    │       │   └── GlobalHooks.java
    │       └── context/
    │           └── ScenarioContext.java            ← @Component + @ScenarioScope
    └── resources/
        ├── features/
        │   ├── auth/
    │   │   └── login.feature
        │   └── payment/
        │       └── checkout.feature
        ├── cucumber.properties
        └── application-test.yml
```

### Multi-module Maven layout
```
parent-pom.xml
├── app/                               ← production code module
│   └── src/main/...
└── integration-tests/                 ← dedicated test module
    ├── pom.xml
    └── src/test/...                   ← ALL test artefacts go here only
```
**Rule:** In multi-module projects, always identify the test module first (Phase 0.3)
and scope every file write to that module's `src/test/` subtree.

---

## Appendix C — Cucumber Version Compatibility Matrix

| Feature / Class                    | Cucumber 6                         | Cucumber 7                              |
|------------------------------------|------------------------------------|-----------------------------------------|
| Context config import              | `io.cucumber.spring`               | `io.cucumber.spring` (same)             |
| Runner annotation                  | `@RunWith(Cucumber.class)` (JUnit 4) | `@Suite` + `@SelectClasses` (JUnit 5) |
| Options                            | `@CucumberOptions(...)`            | `@ConfigurationParameter(...)`          |
| Step imports                       | `io.cucumber.java.en.*`            | `io.cucumber.java.en.*` (same)          |
| Hooks import                       | `io.cucumber.java`                 | `io.cucumber.java` (same)               |
| `@BeforeAll` / `@AfterAll`         | Not available                      | `io.cucumber.java.BeforeAll`            |
| Kotlin step style (preferred)      | `io.cucumber.java8.En` (lambda)    | `io.cucumber.kotlin.*` (annotation)     |
| Row-level tags in Examples         | Available                          | Available                               |
| `Rule:` keyword support            | Available                          | Available                               |
| DocString type                     | `String`                           | `String` or `io.cucumber.docstring.DocString` |
| `@ScenarioScope`                   | `org.springframework.web.context.annotation` | Same                         |

---

*End of automation-skills.md — Version 2.0 (fully hardened)*
