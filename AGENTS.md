# AGENTS.md

This file is the operational handbook for AI coding agents and developers working in this repository.

It does not define architecture. It explains how to work safely within the architecture.

## Documentation Hierarchy

```text
AGENTS.md
        ↓
docs/PROJECT_OPERATING_SPECIFICATION.md
        ↓
Architecture Documents
        ↓
Implementation
```

- **AGENTS.md** explains how agents work.
- **Project Operating Specification (POS)** defines how the system works.
- **Architecture documents** explain detailed technical design decisions.
- **Source code** implements those decisions.

## Mission

Build and evolve the AI Video Translator as a clean, modular, provider-driven system where infrastructure can change without forcing changes to Domain or Application layers.

The architecture must preserve:

- Clean Architecture
- Feature-based modules
- Provider Pattern
- Dependency Inversion
- Stable data contracts
- Replaceable infrastructure

## Required Reading Order

Before modifying code, every agent must read documents in this order:

1. `AGENTS.md`
2. `docs/PROJECT_OPERATING_SPECIFICATION.md`
3. Architecture documents, if present
4. Relevant source code and tests

If documents conflict, follow the nearest/highest-authority document in this order and document the conflict.

## Working Principles

- Understand before modifying.
- Preserve architecture before optimizing code.
- Prefer extension over modification.
- Reuse existing abstractions before adding new ones.
- Create ports before providers.
- Do not bypass providers.
- Do not introduce infrastructure dependencies into Domain or Application.
- Do not hard-code a fixed processing pipeline.
- Keep modules loosely coupled.
- Keep interfaces stable.
- Keep data contracts backward compatible.
- Add the smallest change that satisfies the task.

## Development Workflow

For every feature or change:

1. Understand the problem and current design.
2. Check existing entities, ports, providers, use cases, and orchestrator flow.
3. Reuse existing abstractions where possible.
4. If a new external tool is needed, create or update a Port first.
5. Implement concrete providers only in `src/infrastructure/`.
6. Wire dependencies only from `src/interfaces/` or composition roots.
7. Add or update tests.
8. Update documentation only when the behavior, contract, or operation changes.
9. Verify tests and basic CLI flow before finishing.

## Architectural Rules

The authoritative architecture rules are defined in `docs/PROJECT_OPERATING_SPECIFICATION.md`.

Summary:

- `src/domain/` contains entities and validation only.
- `src/application/` contains use cases, ports, and orchestration policy.
- `src/infrastructure/` contains provider implementations and third-party integrations.
- `src/interfaces/` contains CLI/API entry points and dependency wiring.
- Domain must not import Application, Infrastructure, or Interfaces.
- Application must not import Infrastructure or Interfaces.
- Infrastructure may import Application ports and Domain entities.
- Interfaces may import all layers for composition, parsing, and serialization.

Forbidden examples:

- Calling FFmpeg inside a use case.
- Importing `faster_whisper`, `openai`, `paddleocr`, or `supabase` in Domain/Application.
- Making Domain entities aware of provider configuration.
- Putting business orchestration inside a provider.
- Adding a feature by hard-coding one pipeline branch for one provider.

Required examples:

- Use dependency injection.
- Add interfaces before implementations.
- Keep providers replaceable.
- Keep use cases small and focused.
- Write tests for validation, use cases, and orchestrator behavior.

## Code Quality Rules

- Keep changes small and focused.
- Prefer clear naming over comments.
- Avoid duplicated logic across providers and use cases.
- Avoid broad abstractions with only one implementation unless they define a required port.
- Do not add dependencies unless the standard library or existing code cannot solve the problem.
- Do not hide provider failures; wrap them in project exceptions where appropriate.
- Maintain type hints on public functions and class methods.
- Preserve existing public interfaces unless the task explicitly requires a contract change.

## Testing Rules

- Unit tests should use mock providers by default.
- Real providers require integration tests and must not run by default if they need heavy models, network access, API keys, or GPU.
- Orchestrator tests must cover feature combinations, including skipped features.
- Domain tests must cover invalid data and boundary cases.
- CLI behavior should be smoke-tested when changing composition, provider selection, or arguments.

## Before Finishing Any Task

Checklist:

- Architecture preserved?
- Domain/Application free from infrastructure imports?
- Interfaces stable or deliberately versioned?
- Data contracts backward compatible?
- Tests added or updated?
- Tests passing?
- Basic CLI flow still works when relevant?
- Documentation updated only where necessary?
- No unnecessary dependencies introduced?
- No secrets, API keys, logs, or generated artifacts committed?

## When Unsure

Agents must not silently make architectural assumptions.

When uncertain:

- Pause and identify the ambiguity.
- Prefer asking for clarification over inventing policy.
- Document trade-offs if multiple valid approaches exist.
- Choose the smallest reversible change only when the task is clear.
- Do not weaken architecture to make implementation faster.

## Governance Principle

The project evolves by changing providers, features, and composition — not by breaking the architecture.

If a change requires Domain or Application to know about a specific infrastructure tool, the design is wrong and must be revisited.
