//! Post-pass over automaton + merge diagnostics: drop issues that the regex layer
//! (or merged output) already resolved at the reported byte index.

use crate::enums::token_category::TokenCategory;
use crate::models::lexical_issue::LexicalIssue;
use crate::models::token_model::TokenStruct;

fn issue_superseded_by_regex_or_merge(
    issue: &LexicalIssue,
    raw_merged: &[TokenStruct],
    regex_list: &[TokenStruct],
) -> bool {
    if issue.message.contains("unterminated") {
        return false;
    }
    let Some(i) = issue.char_index else {
        return false;
    };
    let known_raw =
        i < raw_merged.len() && !matches!(raw_merged[i].category, TokenCategory::Unknown);
    let known_regex =
        i < regex_list.len() && !matches!(regex_list[i].category, TokenCategory::Unknown);
    known_raw || known_regex
}

/// Merges automaton and merge-phase issues, removes false positives when the regex
/// or merged grid already classifies the byte, then sorts and deduplicates.
pub fn consolidate_lexical_issues(
    automata_issues: Vec<LexicalIssue>,
    merge_issues: Vec<LexicalIssue>,
    raw_merged: &[TokenStruct],
    regex_list: &[TokenStruct],
) -> Vec<LexicalIssue> {
    let automata_issues: Vec<LexicalIssue> = automata_issues
        .into_iter()
        .filter(|issue| !issue_superseded_by_regex_or_merge(issue, raw_merged, regex_list))
        .collect();

    let merge_issues: Vec<LexicalIssue> = merge_issues
        .into_iter()
        .filter(|issue| !issue_superseded_by_regex_or_merge(issue, raw_merged, regex_list))
        .collect();

    let mut lexical_issues = automata_issues;
    lexical_issues.extend(merge_issues);
    lexical_issues.sort();
    lexical_issues.dedup_by(|a, b| {
        a.lexeme == b.lexeme && a.message == b.message && a.char_index == b.char_index
    });
    lexical_issues
}
