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

use serde::de::DeserializeOwned;
use thiserror::Error;

use crate::conversation::{Message, Role};
use crate::error::AgentError;
use crate::llm::{LlmClient, LlmResponse, Usage};
use crate::tool::ToolDef;

/// Typed structured result plus provider-reported usage.
///
/// Usage stays attached to the call boundary even when a caller does
/// not need it. This avoids forcing evidence- or cost-sensitive
/// consumers to drop metadata merely because the first Zirkel callers
/// only needed the value.
#[derive(Debug)]
pub struct StructuredOutput<T> {
    pub value: T,
    pub usage: Option<Usage>,
}

/// Failure modes for one structured-output model call.
#[derive(Debug, Error)]
pub enum StructuredOutputError {
    #[error("LLM call failed: {0}")]
    Llm(#[source] AgentError),
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

/// Parse one already-received model response through the structured-output contract.
///
/// Keeping response admission separate from provider I/O makes the boundary testable
/// without a live provider and preserves the same checks when the transport later gains
/// provider-native JSON/schema modes.
fn admit_structured_response<T: DeserializeOwned>(
    response: LlmResponse,
    expected_tool: &str,
    usage: Option<Usage>,
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
            let value = serde_json::from_str::<T>(&call.arguments).map_err(|e| {
                StructuredOutputError::ParseArguments {
                    tool: expected_tool.to_string(),
                    error: e.to_string(),
                    raw: call.arguments.clone(),
                }
            })?;
            Ok(StructuredOutput { value, usage })
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
    let (response, usage) = llm
        .complete(messages, &[output_tool], api_key)
        .await
        .map_err(StructuredOutputError::Llm)?;

    admit_structured_response(response, &tool_name, usage)
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

    #[test]
    fn admits_expected_tool_call_and_parses_typed_value() {
        let output = admit_structured_response::<FixtureResult>(
            LlmResponse::ToolCalls(vec![tool_call(
                "emit_fixture",
                r#"{"answer":"yes","count":2}"#,
            )]),
            "emit_fixture",
            None,
        )
        .unwrap();

        assert_eq!(
            output.value,
            FixtureResult {
                answer: "yes".to_string(),
                count: 2,
            }
        );
        assert!(output.usage.is_none());
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
        )
        .unwrap_err();

        assert!(matches!(
            err,
            StructuredOutputError::ParseArguments { ref tool, ref raw: seen, .. }
                if tool == "emit_fixture" && seen == raw
        ));
    }
}
