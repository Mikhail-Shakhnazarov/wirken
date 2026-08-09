# Structured model output

**Branch:** `development/structured-output`  
**Status:** development construction; draft PR #2; repository CI/build not yet observed from the current research runtime

## Why this branch exists

Zirkel had a useful typed-output mechanism hidden inside `crates/zirkel/src/synthetic_tool.rs`: present one schema-bearing tool definition, ask the model to emit that shape, and deserialize the returned arguments into a caller-selected Rust type.

That file explicitly described the mechanism as a local stopgap and said another consumer should trigger extraction rather than duplication. Current `main` had already accumulated three Zirkel consumers before an additional estate consumer needed the same operation.

This branch extracts the transport-level relation and leaves Zirkel-specific schemas in Zirkel.

## Boundary

The generic API is `wirken_agent::structured_output`.

```text
caller-owned messages
+
caller-owned ToolDef result schema
+
existing LlmClient provider dispatch
→
StructuredOutput<T>
```

The tool definition is an output carrier only. The generic structured-output path never dispatches or executes it.

The current cross-provider transport is tool calling because that capability already exists across Wirken's provider adapters. Callers do not depend on that transport choice. Provider-native JSON/schema modes can replace it incrementally without changing the caller contract.

## Result evidence

`StructuredOutput<T>` carries the typed value, provider-reported usage when available, and a `StructuredAttemptReceipt`.

The receipt binds one physical attempt to:

- locally minted attempt id;
- provider and model;
- digest of the full `LlmConfig`;
- digest of the normalized message slice handed to `LlmClient`;
- digest of the schema-bearing `ToolDef`;
- provider-returned tool-call id;
- digest of the raw JSON argument bytes before typed deserialization.

This is an execution receipt, not a domain object identity. A caller's work/basis/requirement/procedure identifiers remain separate and should be retained alongside it when they matter.

## Zirkel migration

Current branch consumers:

- `crates/zirkel/src/llm_score.rs`;
- `crates/zirkel/src/themes.rs`;
- `crates/zirkel/src/perspectives.rs`.

`crates/zirkel/src/synthetic_tool.rs` now keeps only Zirkel-specific schema definitions and typed argument structures.

## What this does not claim

This branch does not create a new agent runtime, conversation model, audit ontology, or provider abstraction. It does not make a model result authoritative merely because deserialization succeeds.

The attempt receipt binds what this direct one-shot boundary materialized and received. It does not automatically provide every stronger property of the full Agent SessionLog path, such as hash-chain position, context-fit reconstruction from earlier session events, credential-slot attribution, or all cost/latency fields.

## Validation pressure

The new module contains pure response-admission tests for:

- expected schema-channel tool call;
- wrong tool name;
- text fallback refusal;
- empty response refusal;
- typed argument mismatch;
- input-message digest sensitivity;
- returned raw-argument digest binding.

This document records construction, not successful execution. Draft PR #2 exists so the branch can receive ordinary repository format/clippy/test pressure before merge.
