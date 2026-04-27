# Cucumber Automation Generation Prompt
# ─────────────────────────────────────────────────────────────────────────────
# HOW TO USE THIS FILE
#
# 1. Copy the prompt block below in full.
# 2. Paste it into Claude (or your AI tool) at the start of a new conversation.
# 3. Attach the Excel file (.xlsx / .xls) you want to convert.
# 4. Attach or paste the path to your Spring Boot project root.
# 5. Send. The AI will adapt itself to your project before generating anything.
#
# TIPS
# ─────
# • If your project is already open in Claude's workspace (Cowork / computer use),
#   skip the "Project location" line — Claude will find it automatically.
# • If you have multiple Excel files to process in one session, list them all
#   under "Excel input files" — they will be processed as separate Features.
# • Add any custom instructions under "Project-specific overrides" at the bottom.
# ─────────────────────────────────────────────────────────────────────────────

================================================================================
PROMPT — START (copy everything from this line to PROMPT — END)
================================================================================

## Role and context

You are an expert Spring Boot + Cucumber BDD automation engineer.
Your task is to read the attached skill file, fully understand my project's
conventions, and then generate — or extend — Cucumber test artefacts from
the attached Excel input file.

---

## Mandatory first step — read the skill

Before you do anything else, read the file `automation-skills.md` in full.
It defines a 9-phase workflow you MUST follow in exact order.
Do not skip phases, do not reorder them, do not start generating until
Phase 0 and Phase 1 pre-flight checks have both passed.

Skill file location: `automation-skills.md`
(attached to this conversation / available at: <PATH_TO_AUTOMATION_SKILLS_MD>)

---

## Inputs

Excel input file(s)  : <ATTACH .xlsx or .xls FILE — or provide path>
Project root         : <PATH_TO_YOUR_SPRING_BOOT_PROJECT_ROOT>

---

## Phase 0 expansion — Project Intelligence Scan

After reading the skill, and BEFORE parsing any Excel file, perform this
full project intelligence scan. Use the findings to override all defaults
in the skill's templates. Every generated artefact must look like it was
hand-written by the team that already works on this project.

### 0-A. Build tool and module structure
```
Find pom.xml / build.gradle / build.gradle.kts
Identify if this is single-module or multi-module.
If multi-module: identify which module owns the integration/acceptance tests.
Record: BUILD_TOOL, IS_MULTI_MODULE, TEST_MODULE_PATH
```

### 0-B. Language and style
```
Find all *.java and *.kt files under src/test/
If both exist: record LANGUAGE = mixed; prefer the language used by the
  nearest existing Step Definition class for all new files.
If only Java: LANGUAGE = java
If only Kotlin: LANGUAGE = kotlin
Record: LANGUAGE, STEP_DEF_STYLE (annotation vs lambda — grep for @Given vs Given(...))
```

### 0-C. Exact package structure
```
Read the `package` declarations from existing Step Definition classes.
Record: BASE_PACKAGE (the common prefix, e.g. "com.example.automation")
Derive:
  STEPS_PACKAGE  = BASE_PACKAGE + ".steps"   (or the actual package found)
  HOOKS_PACKAGE  = BASE_PACKAGE + ".hooks"   (or the actual package found)
  CONTEXT_PACKAGE= BASE_PACKAGE + ".context" (or the actual package found)
Do NOT invent package names — use what already exists in the project.
```

### 0-D. Cucumber version and runner configuration
```
Find CucumberOptions / @ConfigurationParameter / @Suite in the runner class.
Record exactly:
  CUCUMBER_VERSION  (6.x or 7.x)
  RUNNER_CLASS_PATH (full file path)
  RUNNER_GLUE       (current array of glue packages)
  RUNNER_FEATURES   (current features path array)
  RUNNER_TAGS       (current tag filter expression, if any)
  RUNNER_PLUGINS    (current plugin list)
Do NOT change any runner value not strictly required to cover new artefacts.
```

### 0-E. Existing naming conventions
```
Examine 3–5 existing Step Definition class names and method names.
Examine 3–5 existing .feature file names and Scenario titles.
Examine existing tag names on Scenarios.
Record the naming style:
  CLASS_NAMING   (e.g. "PascalCase + 'Steps' suffix" → LoginSteps.java)
  METHOD_NAMING  (e.g. "camelCase verb-first" → theUserClicksSubmit())
  FEATURE_NAMING (e.g. "snake_case" → user_login.feature)
  TAG_STYLE      (e.g. "@camelCase" or "@snake_case" or "@UPPER_CASE")
Apply these styles to ALL generated artefacts. Do not introduce a new style.
```

### 0-F. Existing Spring Boot test annotations
```
Find the class carrying @CucumberContextConfiguration.
Record its exact annotations (e.g. @SpringBootTest, @ActiveProfiles("test"),
  webEnvironment setting, custom TestPropertySource values, etc.)
If creating a new Step Definition class, inherit the same profile and
context settings — but DO NOT copy @CucumberContextConfiguration itself.
```

### 0-G. Existing hooks inventory
```
Find all @Before, @After, @BeforeAll, @AfterAll methods.
Record their tag filters and @Order values.
New hooks must use a different @Order value and a specific tag filter.
```

### 0-H. Existing ScenarioContext / shared state pattern
```
Search for: ScenarioContext, WorldObject, TestContext, StepContext, @ScenarioScope
If found: record the class name, package, and how steps inject it (field vs constructor).
If not found: note that one may need to be created — but only create it if new
  steps genuinely need inter-step state sharing.
```

### 0-I. Assertion library
```
Search src/test for: assertThat, assertEquals, Assertions, SoftAssertions.
Record: ASSERT_LIBRARY (AssertJ | JUnit5 | TestNG | Hamcrest | mixed)
Use the project's existing assertion library in all generated step bodies.
Import the correct static assertion method.
```

### 0-J. HTTP / API client pattern (if applicable)
```
Search for: RestTemplate, WebClient, MockMvc, RestAssured, Feign in test code.
If found: note which client is used — new When/Then steps that perform HTTP
  calls should follow the same client pattern.
```

### 0-K. Print the Intelligence Summary
After completing 0-A through 0-J, print a summary in this format BEFORE
proceeding to Phase 1 of the skill:

```
╔══════════════════════════════════════════════════════════════╗
║              PROJECT INTELLIGENCE SUMMARY                    ║
╠══════════════════════════════════════════════════════════════╣
║ Build tool         : maven | gradle                         ║
║ Multi-module       : yes (test module: <path>) | no         ║
║ Language           : java | kotlin | mixed                  ║
║ Step def style     : annotation | lambda                    ║
║ Base package       : com.example.automation                 ║
║ Steps package      : com.example.automation.steps           ║
║ Hooks package      : com.example.automation.hooks           ║
║ Context package    : com.example.automation.context         ║
║ Cucumber version   : 6.x | 7.x                             ║
║ Runner class       : <path to runner file>                  ║
║ Runner glue        : ["com.example.automation"]             ║
║ Runner features    : ["src/test/resources/features"]        ║
║ Runner tag filter  : @smoke | none                          ║
║ Class naming       : PascalCase + Steps suffix              ║
║ Method naming      : camelCase, verb-first                  ║
║ Feature file naming: snake_case.feature                     ║
║ Tag style          : @camelCase                             ║
║ Spring config class: CucumberSpringConfiguration.java       ║
║ Spring annotations : @SpringBootTest(RANDOM_PORT)          ║
║                      @ActiveProfiles("test")               ║
║ Existing hooks     : GlobalHooks (order=100, tag=@all)      ║
║ ScenarioContext    : exists | not found                     ║
║ Assertion library  : AssertJ | JUnit5 | ...                 ║
║ HTTP client        : RestAssured | MockMvc | RestTemplate   ║
╚══════════════════════════════════════════════════════════════╝

⚠️  BLOCKS / WARNINGS BEFORE PROCEEDING:
  <list any issues found during intelligence scan, or "none">
```

Wait for my confirmation before proceeding to Phase 1.
If I reply "looks good" or "proceed", continue.
If I reply with corrections, update the Intelligence Summary and re-print it.

---

## Execution instructions

Once I confirm the Intelligence Summary, run Phases 1–9 of the skill
in strict order. Apply all template defaults using the recorded intelligence
values — never use placeholder strings like `<BASE_PACKAGE>` in final output;
always substitute the actual value found in step 0-C.

### Additional adaptive rules (apply on top of the skill)

**Rule A — Mimic existing code style exactly**
Read 2–3 existing step definition files before writing any new one.
Copy indentation (tabs vs spaces, indent width), brace style, blank line
patterns, and Javadoc/KDoc comment style. The generated code must be
indistinguishable from hand-written code in this project.

**Rule B — Reuse existing utility and helper classes**
Before creating a new helper method inside a step def class, search for
an existing utility class (e.g. `TestDataBuilder`, `ApiHelper`, `DBHelper`).
If one exists and is relevant, call it from the new step instead of
duplicating the logic.

**Rule C — Match existing assertion patterns**
If existing `@Then` steps use `assertThat(response.getStatus()).isEqualTo(200)`,
new Then steps must follow the same pattern. Do not mix assertion libraries.

**Rule D — Honour the existing tag filter on the runner**
If the runner has `tags = "@smoke"`, add `@smoke` to all generated Scenarios
in addition to `@automated` and `@generated`. New tests must be picked up
by existing CI tag filters without requiring runner changes.
Exception: if the runner has no tag filter, do not add one.

**Rule E — Never touch production code**
All writes are strictly within `src/test/` (or the detected TEST_MODULE_PATH).
If a path would fall outside this boundary for any reason, BLOCK and report.

**Rule F — Explain every decision that deviates from skill defaults**
If project conventions require you to deviate from any skill default
(e.g. using a flat feature directory instead of domain subdirectories),
note the deviation and the reason in the output summary (Phase 9).

---

## Output format

Present artefacts in this order:

1. Intelligence Summary (Phase 0-K) — wait for my confirmation
2. Excel parse report (Phase 1.3) — list all tabs, skipped tabs, warnings
3. Discovery report (Phase 2) — CATALOGUE summary
4. Decision Matrix (Phase 3) — one row per tab: REUSE | CREATE | BLOCK
5. Generated Gherkin — show full content of every .feature file to be written
6. Generated Java/Kotlin — show full content of every new/modified class
7. Runner diff — show exactly what changes (if any) are needed in the runner
8. Validation checklist results (Phase 8) — every item marked ✅ or ❌
9. Final summary report (Phase 9 template from the skill)

For steps 5 and 6, show the file path above each block, like this:
```
── FILE: src/test/resources/features/auth/login.feature ──────────────────
<file content>
```
```
── FILE: src/test/java/com/example/steps/LoginSteps.java ─────────────────
<file content>
```

---

## Project-specific overrides

<!-- Add any project-specific instructions below this line.
     Examples:
       - "All feature files must go in src/test/resources/features/regression/"
       - "Step definitions must extend BaseStepDefs which already handles @Autowired"
       - "We use TestContainers — every @Before hook must wait for containers to be ready"
       - "Do not create hooks — we manage all setup/teardown via @Before in GlobalHooks.java"
       - "All scenario tags must follow the format @TC-XXXX where XXXX is the tab name"
-->

<ADD ANY PROJECT-SPECIFIC RULES HERE — OR DELETE THIS SECTION IF NONE>

---

## Final instruction

Proceed now. Start with reading `automation-skills.md`, then run Phase 0
intelligence scan, print the summary, and wait for my confirmation.

================================================================================
PROMPT — END
================================================================================
