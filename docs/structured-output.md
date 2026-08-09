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

`StructuredOutput<T>` carries:

- the typed value;
- the exact raw JSON argument string returned by the model;
- provider-reported usage when available;
- a `StructuredCallReceipt`.

The raw result remains available because deserializing and later reserializing a typed value can preserve meaning while changing exact bytes.

The receipt binds one bounded `LlmClient::complete` call to:

- locally minted call id;
- provider and model;
- digest of the full `LlmConfig`;
- digest of the normalized message slice handed to `LlmClient`;
- digest of the schema-bearing `ToolDef`;
- provider-returned tool-call id;
- digest of the retained raw JSON argument bytes before typed deserialization.

This is **not** a physical transport-attempt receipt. `LlmClient` may retry HTTP 429 responses internally, so one bounded call can contain several HTTP attempts. Per-transport-attempt identity requires a stronger recovery/session evidence contract.

This correction matters because:

```text
bounded structured call
≠
physical HTTP/provider attempt
```

A caller's work/basis/requirement/procedure identifiers remain separate again.

## Zirkel migration

Current branch consumers:

- `crates/zirkel/src/llm_score.rs`;
- `crates/zirkel/src/themes.rs`;
- `crates/zirkel/src/perspectives.rs`.

`crates/zirkel/src/synthetic_tool.rs` now keeps only Zirkel-specific schema definitions and typed argument structures.

## Generic one-shot runner

`crates/agent/examples/structured_once.rs` consumes one provider-neutral content-addressed call manifest without entering the Agent loop.

It verifies the manifest before execution and emits:

- opaque caller metadata;
- typed JSON value;
- exact raw result;
- provider usage;
- `call_receipt`.

The manifest contains no credential.

## What this does not claim

This branch does not create a new agent runtime, conversation model, audit ontology, or provider abstraction. It does not make a model result authoritative merely because deserialization succeeds.

The bounded call receipt does not automatically provide every stronger property of the full Agent SessionLog/recovery path, including per-transport-attempt identity, hash-chain position, context-fit reconstruction from earlier session events, credential-slot attribution, or all cost/latency fields.

## Validation pressure

The new module contains pure response-admission tests for:

- expected schema-channel tool call;
- wrong tool name;
- text fallback refusal;
- empty response refusal;
- typed argument mismatch;
- input-message digest sensitivity;
- exact raw-result retention and digest binding.

The generic runner contains pure manifest-canonicalization/mutation tests.

This document records construction, not successful execution. Draft PR #2 exists so the branch can receive ordinary repository format/clippy/test pressure before merge.
