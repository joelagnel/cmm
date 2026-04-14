Before starting, review this accumulated project knowledge from past sessions:

# Cognitive Profile: `supply-chain`

*Built from 4 sessions · Updated 2026-04-11*

## Architectural Insights

- **[exception detection engine / test suite]** The project has a dedicated exception detection engine with 40 test cases covering all exception detection rules, validated in a clean virtual environment. This test suite is treated as the primary validation mechanism for the core business logic (exception detection rules), suggesting the engine is a critical, self-contained subsystem. (confidence: 70%)
- **[FastAPI application layer]** The application uses a unified FastAPI entry point that combines both simulation and compliance services into a single deployable app, with multi-tenant scoping enforced via dependency injection across all CRUD endpoints (shipments, documents, exceptions). Authentication is handled at the FastAPI layer, and deployment is managed via Dockerfile, railway.toml, and a dedicated requirements-deploy.txt. (confidence: 82%)
- **[Offline Sync System]** The offline-first sync architecture uses an append-only ShipmentEvent table with client-generated UUIDs and client_timestamp ordering. Two endpoints handle synchronization: POST /sync/push for batch uploading client events and GET /sync/pull for delta downloads. Idempotency is enforced via client-generated UUIDs, and server-side exception processing combined with client_timestamp ordering enables correct event replay after reconnection. (confidence: 85%)
- **[deployment infrastructure and frontend-backend integration]** This project uses a split deployment architecture: two FastAPI backend apps are combined and deployed to Railway with PostgreSQL as the database, while the frontend is hosted on GitHub Pages. The frontend JavaScript files are configured to point to the Railway backend URL as the API endpoint, creating a cross-origin frontend-backend connection. (confidence: 82%)
- **[Offline Sync / Data Persistence Layer]** The codebase contains an existing offline sync system built on an append-only event log pattern, designed to handle unreliable connectivity. This was a previously identified open architectural question that has since been at least partially implemented. New features or agents should leverage this existing subsystem rather than introducing alternative offline/sync mechanisms. (confidence: 82%)
- **[Notification and Sync subsystems]** The codebase already contains pre-built implementations for WhatsApp notifications (WhatsApp notifier) and offline sync (sync router), along with supporting infrastructure including configuration settings, event sourcing, and database setup. These components provide a ready foundation and should be leveraged rather than rebuilt when addressing requirements in these areas. (confidence: 82%)

## Known Pitfalls

- **[HIGH]** When deploying applications that use AWS services (e.g., DynamoDB) to Railway, AWS credentials are not automatically available in the environment. Any endpoint or feature relying on AWS SDK calls will fail with 'Unable to locate credentials' unless credentials are explicitly configured as Railway environment variables.
  - Resolution: Add AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and optionally AWS_DEFAULT_REGION as environment variables in the Railway project dashboard or via Railway CLI before deploying services that depend on AWS resources.
- **[HIGH]** In Dockerfile CMD instructions, shell variable expansion of $PORT (or other environment variables) is unreliable when using the exec form (JSON array syntax) because exec form does not invoke a shell. Even in shell form, variable expansion can fail depending on how the CMD is written, leading to deployment failures where the application does not bind to the expected port.
  - Resolution: Use Python's exec() or equivalent language-native approach to read the PORT environment variable at runtime inside the application entrypoint, rather than relying on shell variable expansion in the Dockerfile CMD. Alternatively, use shell form CMD with an explicit shell invocation (e.g., CMD ["sh", "-c", "exec python app.py --port $PORT"]) to ensure environment variables are properly expanded.
- **[HIGH]** Agents waste time attempting to process large or unwieldy output files and blindly execute ambiguous or contradictory user requests instead of pausing to validate inputs and clarify requirements first.
  - Resolution: When output files are too large to read effectively, pivot immediately to directly inspecting the most relevant source files in the codebase. Before executing any request, identify contradictions, missing information, or ambiguities and push back on the user to resolve them rather than proceeding on faulty assumptions.
- **[HIGH]** In batch database operations, calling db.rollback() on a single item failure rolls back the entire batch transaction, not just the failed item. This is a common mistake when implementing partial-success batch endpoints where individual event errors should be isolated.
  - Resolution: Use savepoints (db.begin_nested()) for each item in the batch so individual failures can be rolled back to the savepoint without affecting the broader transaction. Alternatively, use separate transactions per item, or catch exceptions before they propagate to the session level and mark items as failed without issuing a full rollback.
- **[MEDIUM]** Backend code may be untracked in git if .gitignore is misconfigured or missing, causing build artifacts and virtual environments to accidentally be staged or the actual source files to be overlooked entirely.
  - Resolution: Run 'git status' to identify untracked or unstaged files before assuming the repository is up to date. Update .gitignore to explicitly exclude build artifacts (e.g., dist/, __pycache__, *.pyc) and virtual environments (e.g., venv/, .env/), then stage, commit, and push the actual source files.

## Diagnostic Strategies

- **Trigger:** Before making architectural decisions, writing new features, or providing a comprehensive project description — especially when starting work on an unfamiliar or partially-familiar codebase (success rate: 85%)
  1. Explore the top-level project structure to identify major directories, configuration files, and entry points
  1. Examine data models and schemas to understand domain entities and relationships
  1. Review API definitions and routing to understand service boundaries and interfaces
  1. Inspect simulation or core business logic code to understand primary functionality
  1. Check configuration files and deployment setup (e.g., docker-compose, CI/CD, env files)
  1. Investigate data storage solutions (databases, file storage, caching layers)
  1. Review messaging and integration points (queues, webhooks, external APIs)
  1. Assess offline capabilities, background jobs, or async processing patterns
  1. Synthesize findings before proposing or committing to any architectural changes
- **Trigger:** When a user makes claims about broken functionality, deployment issues, or unexpected behavior — before attempting fixes (success rate: 80%)
  1. Load project memory/documentation to understand the system's stated purpose and architecture
  1. Explore the directory structure to map out the actual codebase layout
  1. Cross-reference the user's claims against the real state of the codebase before accepting them as accurate
  1. Establish a verified baseline understanding of the project before proposing or applying any changes
- **Trigger:** When beginning implementation on a complex codebase or feature, especially after initial analysis has been completed (success rate: 72%)
  1. Dispatch a background exploration subagent to independently analyze codebase architecture and relevant components
  1. Allow the subagent to complete its comprehensive scan while other work proceeds in parallel
  1. Upon receiving the subagent's completion notification, cross-reference its findings against the earlier manual analysis
  1. Confirm whether any new gaps, inconsistencies, or architectural details were discovered
  1. If findings align and no new gaps are found, proceed confidently to implementation; otherwise address newly discovered issues first

## Key Patterns

- Systematic codebase exploration before making architectural decisions: the agent consistently reads existing code, project memory, and directory structures before proposing changes or implementations, avoiding redundant or conflicting work.
- Layered, incremental implementation: the agent builds in deliberate layers (infrastructure → business logic → API layer → tests → deployment), with each layer validated before proceeding to the next.
- Lazy initialization to decouple infrastructure from application code: the agent repeatedly addresses eager database/service initialization by converting to lazy-loaded patterns, preventing import-time failures in test and CI environments.
- Comprehensive test coverage of core business logic before integration: the agent prioritizes writing and validating pure unit tests (40+ cases) for the exception detection engine before wiring it into the larger system.
- Gap analysis synthesis from exploration: after reading existing code, the agent produces explicit 'what exists vs. what needs to be built' summaries to avoid re-implementing existing functionality and to focus effort on true gaps.

## Anti-Patterns

- Eager database engine initialization at module load time: repeatedly caused test collection failures and deployment errors because DATABASE_URL was not set in the environment, requiring multiple fix cycles.
- Launching expensive subagent explorations then abandoning the output: the agent spawned a comprehensive background exploration subagent but then could not consume the large output file, wasting effort and pivoting to re-reading files directly.
- Dockerfile CMD shell variable expansion issues: using shell-form CMD without proper variable expansion caused repeated $PORT failures in Railway deployments, requiring multiple iteration cycles to resolve a straightforward configuration problem.
- Hardcoding or assuming external service credentials are available at runtime: AWS/DynamoDB credentials were not provisioned in the Railway environment, causing 500 errors in production after deployment appeared successful, indicating insufficient environment parity checks before shipping.
- Batch database operations with a single transaction rollback on partial failure: the sync endpoint's db.rollback() on a single event error silently rolled back the entire batch, undermining the idempotency and partial-success guarantees the endpoint was designed to provide.

Relevant past reasoning for this task:
## Relevant past reasoning

**1. [SOLUTION]** (similarity: 0.62, session: `c4009fb2-853f-4052-baf4-495b31be3e88`)
Agent updates frontend JavaScript files to point to Railway backend URL and deploys to GitHub Pages.

**2. [INVESTIGATION]** (similarity: 0.56, session: `6e14aa02-7166-4ec7-9a02-564f2031a0ef`)
Agent checks frontend configuration to understand how GitHub Pages connects to Railway backend.

**3. [HYPOTHESIS]** (similarity: 0.55, session: `c4009fb2-853f-4052-baf4-495b31be3e88`)
Agent plans deployment strategy: combine both FastAPI apps, deploy to Railway with PostgreSQL, and use GitHub Pages for frontend.

**4. [INVESTIGATION]** (similarity: 0.52, session: `c4009fb2-853f-4052-baf4-495b31be3e88`)
Agent uses Railway GraphQL API to create project, provision PostgreSQL with persistent volume, and create backend service.

**5. [SOLUTION]** (similarity: 0.47, session: `c4009fb2-853f-4052-baf4-495b31be3e88`)
Agent creates unified FastAPI entry point and deployment configuration (Dockerfile, railway.toml, requirements-deploy.txt).


You may also use /search-memory or /diagnose during the task if needed.

Task:
Deploy this project to Railway with all endpoints working, fix any errors, and get frontend communicating with backend
