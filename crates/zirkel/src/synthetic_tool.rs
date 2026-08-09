//! Zirkel-specific schema carriers for structured model output.
//!
//! Zirkel originally owned both these schema definitions and the
//! transport that forced models to return them through synthetic tool
//! calls. The transport is now reusable at
//! [`wirken_agent::structured_output`]. This module retains only the
//! domain-specific output names, JSON schemas, and typed argument
//! structures.
//!
//! The `zirkel_`-prefixed tool names are response-channel identifiers;
//! the orchestrator never executes them. Callers outside Zirkel should
//! define their own result contract and use the generic structured
//! output boundary rather than importing these schemas.

use serde::Deserialize;
use wirken_agent::tool::ToolDef;

/// Tool def for `zirkel_score_candidate` — used by the LLM relevance
/// scorer. Output type: [`ScoreCandidateArgs`].
pub fn score_candidate_tool() -> ToolDef {
    ToolDef {
        name: "zirkel_score_candidate".to_string(),
        description: "Return a structured relevance score for the candidate item against the user's interests. \
                      You MUST call this tool with the score, why-surfaced rationale, and matched keyword. \
                      Do not respond with text.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Relevance score 0–100. 0 = irrelevant. 50 = matches a user interest broadly. 100 = directly addresses a user keyword in a substantive way."
                },
                "why_surfaced": {
                    "type": "string",
                    "description": "One-line rationale (≤140 chars). Cite the matched user interest by name and how the candidate's content relates to it."
                },
                "matched_keyword": {
                    "type": "string",
                    "description": "The single user keyword that best characterizes the match. Must be one of the user keywords provided in the prompt."
                }
            },
            "required": ["score", "why_surfaced", "matched_keyword"]
        }),
    }
}

/// Output shape for [`score_candidate_tool`].
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ScoreCandidateArgs {
    pub score: u32,
    pub why_surfaced: String,
    pub matched_keyword: String,
}

/// Tool def for `zirkel_name_theme` — used by the theme naming pass.
/// Output type: [`NameThemeArgs`].
pub fn name_theme_tool() -> ToolDef {
    ToolDef {
        name: "zirkel_name_theme".to_string(),
        description: "Return a 2–5 word theme name for a cluster of related candidates. \
                      You MUST call this tool. Do not respond with text or quotes."
            .to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "2–5 word theme name. Examples: 'FTC enforcement', 'EU AI Act', 'biometric privacy in employment'. No cluster ids, no jargon, no quotes."
                }
            },
            "required": ["name"]
        }),
    }
}

/// Output shape for [`name_theme_tool`].
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct NameThemeArgs {
    pub name: String,
}

/// Tool def for `zirkel_emit_perspectives` -- the librarian's
/// perspective-expansion call. Single LLM call: input is the topic
/// plus concatenated section headings from related Wikipedia
/// articles (gathered out-of-band by [`crate::perspectives`]),
/// output is a short list of noun-phrase perspective labels. The
/// labels are ephemeral and never persisted outside the audit
/// chain. Output type: [`EmitPerspectivesArgs`].
pub fn emit_perspectives_tool(max_perspectives: usize) -> ToolDef {
    let cap = max_perspectives.max(1);
    ToolDef {
        name: "zirkel_emit_perspectives".to_string(),
        description: format!(
            "Return up to {cap} short noun-phrase perspective labels for the given topic, \
             each grounded in the supplied section headings. Each label must be a 2-5 word noun \
             phrase that names a distinct angle on the topic. \
             You MUST call this tool. Do not respond with text or quotes."
        ),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "perspectives": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A 2-5 word noun-phrase label naming one angle on the topic."
                    },
                    "maxItems": cap,
                    "description": format!(
                        "Up to {cap} distinct perspective labels. No duplicates. \
                         Examples for topic 'biometric privacy': 'employer surveillance', \
                         'consent frameworks', 'state-level enforcement', 'cross-border transfer'."
                    )
                }
            },
            "required": ["perspectives"]
        }),
    }
}

/// Output shape for [`emit_perspectives_tool`].
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct EmitPerspectivesArgs {
    pub perspectives: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_candidate_tool_def_has_required_fields() {
        let tool = score_candidate_tool();
        assert_eq!(tool.name, "zirkel_score_candidate");
        let required = tool
            .parameters
            .get("required")
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert!(required.contains(&"score".to_string()));
        assert!(required.contains(&"why_surfaced".to_string()));
        assert!(required.contains(&"matched_keyword".to_string()));
    }

    #[test]
    fn name_theme_tool_def_requires_name() {
        let tool = name_theme_tool();
        assert_eq!(tool.name, "zirkel_name_theme");
        let required = tool
            .parameters
            .get("required")
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].as_str().unwrap(), "name");
    }

    #[test]
    fn score_args_round_trip_through_serde() {
        let json = r#"{"score": 75, "why_surfaced": "matched 'BIPA' — biometric enforcement", "matched_keyword": "BIPA"}"#;
        let parsed: ScoreCandidateArgs = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.score, 75);
        assert_eq!(parsed.matched_keyword, "BIPA");
        assert!(parsed.why_surfaced.contains("biometric"));
    }

    #[test]
    fn theme_name_args_round_trip() {
        let json = r#"{"name": "FTC enforcement"}"#;
        let parsed: NameThemeArgs = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.name, "FTC enforcement");
    }

    #[test]
    fn emit_perspectives_tool_def_caps_at_max() {
        let tool = emit_perspectives_tool(4);
        assert_eq!(tool.name, "zirkel_emit_perspectives");
        let max_items = tool
            .parameters
            .pointer("/properties/perspectives/maxItems")
            .and_then(|v| v.as_u64())
            .unwrap();
        assert_eq!(max_items, 4);
        let required = tool
            .parameters
            .get("required")
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].as_str().unwrap(), "perspectives");
    }

    #[test]
    fn emit_perspectives_args_round_trip() {
        let json = r#"{"perspectives": ["employer surveillance", "consent frameworks"]}"#;
        let parsed: EmitPerspectivesArgs = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.perspectives.len(), 2);
        assert_eq!(parsed.perspectives[0], "employer surveillance");
    }
}
