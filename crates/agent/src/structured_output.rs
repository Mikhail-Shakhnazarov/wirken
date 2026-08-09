//! Provider-neutral structured model output over [`crate::llm::LlmClient`].
//!
//! This module extracts a reusable caller-facing contract from the
//! Zirkel-local synthetic-tool idiom. Callers supply the output schema
//! as a [`ToolDef`] and receive typed arguments rather than inspecting
//! provider response text themselves.
//!
//! The current transport uses the already-supported tool-call path on
//! every provider. That is an implementation detail of this boundary,
//! not a requirement on callers. Provider-native JSON/schema response
//! modes can replace the transport incrementally without changing the
//! caller contract.
//!
//! This module does **not** execute the supplied tool. The tool is an
//! output schema carrier only.

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::conversation::{Message, Role};
use crate::error::AgentError;
use crate::llm::{LlmClient, LlmResponse, Usage};
use crate::tool::ToolDef;

/// Evidence for one bounded [`LlmClient::complete`] invocation that
/// produced a structured result.
///
/// This is deliberately separate from any domain-level basis identity.
/// It is also deliberately **not** called a physical-attempt receipt:
/// `LlmClient` may internally retry HTTP 429 responses, so one bounded
/// completion can contain several transport attempts. This receipt binds
/// the caller-visible structured call and its admitted result. Consumers
/// that need per-transport-attempt identity require the stronger recovery/
/// session audit path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StructuredCallReceipt {
    /// Locally minted identity for this bounded structured call.
    pub call_id: String,
    /// Provider and model from the exact [`crate::llm::LlmConfig`] used.
    pub provider: String,
    pub model: String,
    /// Digest of the complete LLM config. This includes generation
    /// parameters and endpoint configuration in addition to provider/model.
    pub config_sha256: String,
    /// Digest of the normalized message slice handed to [`LlmClient`].
    /// For this one-shot boundary there is no hidden context-fit step:
    /// these are the messages passed directly to the provider adapter.
    pub messages_sha256: String,
    /// Digest of the exact schema-bearing [`ToolDef`] supplied as the
    /// structured response channel.
    pub output_schema_sha256: String,
    /// Provider-returned tool-call identity when structured admission
    /// succeeded.
    pub response_tool_call_id: String,
    /// Digest of the raw JSON argument bytes returned by the model before
    /// typed deserialization.
    pub response_arguments_sha256: String,
}

#[derive(Debug, Clone)]
struct StructuredCallStart {
    call_id: String,
    provider: String,
    model: String,
    config_sha256: String,
    messages_sha256: String,
    output_schema_sha256: String,
}

/// Typed structured result plus the exact raw result bytes, provider-reported
/// usage, and bounded-call evidence.
///
/// Retaining `raw_arguments` matters when later audit/replay needs the model's
/// exact returned JSON rather than a reserialization of `value`, which may be
/// semantically equivalent but byte-different.
#[derive(Debug)]
pub struct StructuredOutput<T> {
    pub value: T,
    pub raw_arguments: String,
    pub usage: Option<Usage>,
    pub receipt: StructuredCallReceipt,
}

/// Failure modes for one structured-output model call.
#[derive(Debug, Error)]
pub enum StructuredOutputError {
    #[error("LLM call failed: {0}")]
    Llm(#[source] AgentError),
    #[error("could not serialize structured-output request evidence: {0}")]
    Evidence(String),
    #[error("expected structured output via '{expected}' but the model responded with text: {text}")]
    ExpectedToolCallGotText { expected: String, text: String },
    #[error("LLM returned an empty response")]
    EmptyResponse,
    #[error("LLM made tool calls but none were to '{expected}'; calls: {actual:?}")]
    ToolNotCalled {
        expected: String,
        actual: Vec<String>,
    },
    #[error("could not parse '{tool}' arguments as the expected shape: {error}; raw: {raw}")]
    ParseArguments {
        tool: String,
        error: String,
        raw: String,
    },
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for &byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("sha256:{}", hex(digest.as_ref()))
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String, StructuredOutputError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|e| StructuredOutputError::Evidence(e.to_string()))?;
    Ok(sha256_bytes(&bytes))
}

fn call_start(
    llm: &LlmClient,
    messages: &[Message],
    output_tool: &ToolDef,
) -> Result<StructuredCallStart, StructuredOutputError> {
    Ok(StructuredCallStart {
        call_id: format!("structured-{}", uuid::Uuid::new_v4()),
        provider: llm.config().provider.clone(),
        model: llm.config().model.clone(),
        config_sha256: sha256_json(llm.config())?,
        messages_sha256: sha256_json(&messages)?,
        output_schema_sha256: sha256_json(output_tool)?,
    })
}

/// Parse one already-received model response through the structured-output contract.
///
/// Keeping response admission separate from provider I/O makes the boundary testable
/// without a live provider and preserves the same checks when the transport later gains
/// provider-native JSON/schema modes.
fn admit_structured_response<T: DeserializeOwned>(
    response: LlmResponse,
    expected_tool: &str,
    usage: Option<Usage>,
    start: StructuredCallStart,
) -> Result<StructuredOutput<T>, StructuredOutputError> {
    match response {
        LlmResponse::ToolCalls(calls) => {
            let actual: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();
            let call = calls
                .iter()
                .find(|c| c.name == expected_tool)
                .ok_or_else(|| StructuredOutputError::ToolNotCalled {
                    expected: expected_tool.to_string(),
                    actual,
                })?;
            let raw_arguments = call.arguments.clone();
            let value = serde_json::from_str::<T>(&raw_arguments).map_err(|e| {
                StructuredOutputError::ParseArguments {
                    tool: expected_tool.to_string(),
                    error: e.to_string(),
                    raw: raw_arguments.clone(),
                }
            })?;
            let receipt = StructuredCallReceipt {
                call_id: start.call_id,
                provider: start.provider,
                model: start.model,
                config_sha256: start.config_sha256,
                messages_sha256: start.messages_sha256,
                output_schema_sha256: start.output_schema_sha256,
                response_tool_call_id: call.id.clone(),
                response_arguments_sha256: sha256_bytes(raw_arguments.as_bytes()),
            };
            Ok(StructuredOutput {
                value,
                raw_arguments,
                usage,
                receipt,
            })
        }
        LlmResponse::Text(text) => Err(StructuredOutputError::ExpectedToolCallGotText {
            expected: expected_tool.to_string(),
            text,
        }),
        LlmResponse::Empty => Err(StructuredOutputError::EmptyResponse),
    }
}

/// Run one structured-output completion over an already-materialized
/// message slice.
///
/// `output_tool` is used only as a schema-bearing response channel.
/// The returned tool arguments are parsed into `T`; no tool is
/// dispatched or granted effect authority by this function.
pub async fn complete_structured<T: DeserializeOwned>(
    llm: &LlmClient,
    api_key: Option<&str>,
    messages: &[Message],
    output_tool: ToolDef,
) -> Result<StructuredOutput<T>, StructuredOutputError> {
    let tool_name = output_tool.name.clone();
    let start = call_start(llm, messages, &output_tool)?;
    let (response, usage) = llm
        .complete(messages, &[output_tool], api_key)
        .await
        .map_err(StructuredOutputError::Llm)?;

    admit_structured_response(response, &tool_name, usage, start)
}

/// Convenience wrapper for the common bounded system+user call shape.
///
/// Keeping message construction here prevents each structured-output
/// consumer from reimplementing the same two-message transport while
/// still allowing [`complete_structured`] callers to supply a richer
/// exact message sequence when needed.
pub async fn complete_structured_prompt<T: DeserializeOwned>(
    llm: &LlmClient,
    api_key: Option<&str>,
    system_prompt: &str,
    user_prompt: &str,
    output_tool: ToolDef,
) -> Result<StructuredOutput<T>, StructuredOutputError> {
    let messages = vec![
        Message {
            role: Role::System,
            content: system_prompt.to_string(),
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
        },
        Message {
            role: Role::User,
            content: user_prompt.to_string(),
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
        },
    ];
    complete_structured(llm, api_key, &messages, output_tool).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::ToolCallRequest;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq, Eq)]
    struct FixtureResult {
        answer: String,
        count: u32,
    }

    fn tool_call(name: &str, arguments: &str) -> ToolCallRequest {
        ToolCallRequest {
            id: "call-1".to_string(),
            name: name.to_string(),
            arguments: arguments.to_string(),
        }
    }

    fn start() -> StructuredCallStart {
        StructuredCallStart {
            call_id: "structured-test".to_string(),
            provider: "test-provider".to_string(),
            model: "test-model".to_string(),
            config_sha256: "sha256:config".to_string(),
            messages_sha256: "sha256:messages".to_string(),
            output_schema_sha256: "sha256:schema".to_string(),
        }
    }

    #[test]
    fn admits_expected_tool_call_and_binds_call_evidence() {
        let raw = r#"{"answer":"yes","count":2}"#;
        let output = admit_structured_response::<FixtureResult>(
            LlmResponse::ToolCalls(vec![tool_call("emit_fixture", raw)]),
            "emit_fixture",
            None,
            start(),
        )
        .unwrap();

        assert_eq!(
            output.value,
            FixtureResult {
                answer: "yes".to_string(),
                count: 2,
            }
        );
        assert_eq!(output.raw_arguments, raw);
        assert!(output.usage.is_none());
        assert_eq!(output.receipt.call_id, "structured-test");
        assert_eq!(output.receipt.provider, "test-provider");
        assert_eq!(output.receipt.model, "test-model");
        assert_eq!(output.receipt.response_tool_call_id, "call-1");
        assert_eq!(
            output.receipt.response_arguments_sha256,
            sha256_bytes(raw.as_bytes())
        );
    }

    #[test]
    fn call_evidence_changes_when_messages_change() {
        let llm = LlmClient::new(crate::llm::LlmConfig::ollama("fixture-model")).unwrap();
        let tool = ToolDef {
            name: "emit_fixture".to_string(),
            description: "fixture".to_string(),
            parameters: serde_json::json!({"type":"object"}),
        };
        let messages_a = vec![Message {
            role: Role::User,
            content: "A".to_string(),
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
        }];
        let messages_b = vec![Message {
            role: Role::User,
            content: "B".to_string(),
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
        }];
        let a = call_start(&llm, &messages_a, &tool).unwrap();
        let b = call_start(&llm, &messages_b, &tool).unwrap();
        assert_ne!(a.messages_sha256, b.messages_sha256);
        assert_eq!(a.output_schema_sha256, b.output_schema_sha256);
    }

    #[test]
    fn refuses_tool_calls_that_do_not_include_expected_channel() {
        let err = admit_structured_response::<FixtureResult>(
            LlmResponse::ToolCalls(vec![tool_call(
                "other_tool",
                r#"{"answer":"yes","count":2}"#,
            )]),
            "emit_fixture",
            None,
            start(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            StructuredOutputError::ToolNotCalled { ref expected, ref actual }
                if expected == "emit_fixture" && actual == &vec!["other_tool".to_string()]
        ));
    }

    #[test]
    fn refuses_text_when_structured_output_was_requested() {
        let err = admit_structured_response::<FixtureResult>(
            LlmResponse::Text("plain text".to_string()),
            "emit_fixture",
            None,
            start(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            StructuredOutputError::ExpectedToolCallGotText { ref expected, ref text }
                if expected == "emit_fixture" && text == "plain text"
        ));
    }

    #[test]
    fn refuses_empty_response() {
        let err = admit_structured_response::<FixtureResult>(
            LlmResponse::Empty,
            "emit_fixture",
            None,
            start(),
        )
        .unwrap_err();

        assert!(matches!(err, StructuredOutputError::EmptyResponse));
    }

    #[test]
    fn refuses_arguments_that_do_not_match_typed_contract() {
        let raw = r#"{"answer":"yes","count":"two"}"#;
        let err = admit_structured_response::<FixtureResult>(
            LlmResponse::ToolCalls(vec![tool_call("emit_fixture", raw)]),
            "emit_fixture",
            None,
            start(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            StructuredOutputError::ParseArguments { ref tool, raw: ref seen, .. }
                if tool == "emit_fixture" && seen == raw
        ));
    }
}
