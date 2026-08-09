//! Execute one provider-neutral structured-call manifest without entering the Agent loop.
//!
//! Usage:
//!
//! ```text
//! cargo run -p wirken-agent --example structured_once -- <manifest.json> <llm-config.json> [API_KEY_ENV]
//! ```
//!
//! `API_KEY_ENV` defaults to `WIRKEN_LLM_API_KEY`. The manifest contains no credential.
//! Providers that do not use an API key may leave the variable unset.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use wirken_agent::conversation::Message;
use wirken_agent::llm::{LlmClient, LlmConfig};
use wirken_agent::structured_output::complete_structured;
use wirken_agent::tool::ToolDef;

#[derive(Debug, Deserialize)]
struct StructuredCallManifest {
    schema_version: u32,
    materialization_id: String,
    #[serde(default)]
    metadata: Value,
    messages: Vec<Message>,
    output_tool: ToolDef,
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => serde_json::to_string(v).expect("serializing JSON string cannot fail"),
        Value::Array(values) => {
            let body = values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort();
            let body = keys
                .into_iter()
                .map(|key| {
                    let rendered_key =
                        serde_json::to_string(key).expect("serializing JSON key cannot fail");
                    format!("{rendered_key}:{}", canonical_json(&values[key]))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
    }
}

fn sha256_text(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        hex.push(DIGITS[(byte >> 4) as usize] as char);
        hex.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    format!("sha256:{hex}")
}

fn materialization_core(manifest: &StructuredCallManifest) -> Value {
    serde_json::json!({
        "schema_version": manifest.schema_version,
        "metadata": &manifest.metadata,
        "messages": &manifest.messages,
        "output_tool": &manifest.output_tool,
    })
}

fn verify_materialization(manifest: &StructuredCallManifest) -> Result<()> {
    let actual = sha256_text(&canonical_json(&materialization_core(manifest)));
    if actual != manifest.materialization_id {
        bail!(
            "manifest materialization_id mismatch: declared {}, computed {}",
            manifest.materialization_id,
            actual
        );
    }
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if !(3..=4).contains(&args.len()) {
        bail!(
            "usage: {} <manifest.json> <llm-config.json> [API_KEY_ENV]",
            args.first().map(String::as_str).unwrap_or("structured_once")
        );
    }

    let manifest: StructuredCallManifest = read_json(Path::new(&args[1]))?;
    verify_materialization(&manifest)?;

    let config: LlmConfig = read_json(Path::new(&args[2]))?;
    if !config.tools_enabled {
        bail!(
            "structured_once currently uses the tool-call transport; llm config has tools_enabled=false"
        );
    }

    let key_env = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("WIRKEN_LLM_API_KEY");
    let api_key = std::env::var(key_env).ok();

    let llm = LlmClient::new(config)?;
    let output = complete_structured::<Value>(
        &llm,
        api_key.as_deref(),
        &manifest.messages,
        manifest.output_tool,
    )
    .await?;

    let record = serde_json::json!({
        "schema_version": 1,
        "materialization_id": manifest.materialization_id,
        "metadata": manifest.metadata,
        "value": output.value,
        "raw_arguments": output.raw_arguments,
        "usage": output.usage,
        "call_receipt": output.receipt,
    });
    println!("{}", serde_json::to_string_pretty(&record)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_sorts_object_keys_recursively() {
        let value = serde_json::json!({
            "z": 1,
            "a": {"y": 2, "b": 3},
            "m": [{"d": 4, "c": 5}]
        });
        assert_eq!(
            canonical_json(&value),
            r#"{"a":{"b":3,"y":2},"m":[{"c":5,"d":4}],"z":1}"#
        );
    }

    #[test]
    fn materialization_verification_detects_manifest_mutation() {
        let mut manifest = StructuredCallManifest {
            schema_version: 1,
            materialization_id: String::new(),
            metadata: serde_json::json!({"basis_id": "b"}),
            messages: vec![Message {
                role: wirken_agent::conversation::Role::User,
                content: "hello".to_string(),
                tool_call_id: None,
                tool_name: None,
                tool_calls: None,
            }],
            output_tool: ToolDef {
                name: "emit".to_string(),
                description: "emit".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
        };
        manifest.materialization_id =
            sha256_text(&canonical_json(&materialization_core(&manifest)));
        verify_materialization(&manifest).unwrap();

        manifest.messages[0].content = "changed".to_string();
        assert!(verify_materialization(&manifest).is_err());
    }
}
