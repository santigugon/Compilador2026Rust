use crate::enums::token_category::TokenCategory;
use crate::models::lexical_issue::LexicalIssue;
use crate::models::token_model::TokenStruct;


const RESERVED_AS_KEYWORD: &[&str] = &[
    "and", "break", "continue", "def", "elif", "else", "False", "for", "if", "in", "is",
    "None", "not", "or", "pass", "return", "True", "while",
];

fn is_reserved_identifier_lexeme(word: &str) -> bool {
    RESERVED_AS_KEYWORD.iter().any(|&kw| kw == word)
}

fn reclassify_reserved_identifiers(mut tokens: Vec<TokenStruct>) -> Vec<TokenStruct> {
    for t in &mut tokens {
        if matches!(&t.category, TokenCategory::Identifier) && is_reserved_identifier_lexeme(&t.word)
        {
            t.category = TokenCategory::Keyword;
        }
    }
    tokens
}

pub fn filter_unknowns(final_list: &Vec<TokenStruct>) -> Vec<TokenStruct> {
    let mut result: Vec<TokenStruct> = vec![];

    for token in final_list {
        match token.category {
            TokenCategory::Unknown | TokenCategory::Indentation | TokenCategory::WhiteSpace => {
                continue;
            }
            _ => {
                if let Some(last) = result.last() {
                    if last.word == token.word && last.category == token.category {
                        continue;
                    }
                }
                result.push(token.clone());
            }
        }
    }

    result
}

/// Positions covered by a successful automaton token are still `Unknown` in the per-slot
/// vector except at the lexeme's end index. Only flag runs where the automaton did not
/// claim the character (`positions_list[i] == false`), merged output is `Unknown`, and the
/// regex layer also has no classification at that byte (regex may still match what the DFA
/// rejected).
fn collect_unknown_runs(
    merged: &[TokenStruct],
    input: &str,
    positions_list: &[bool],
    regex_list: &[TokenStruct],
) -> Vec<LexicalIssue> {
    let chars: Vec<char> = input.chars().collect();
    let n = chars
        .len()
        .min(merged.len())
        .min(positions_list.len())
        .min(regex_list.len());
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if positions_list[i] {
            i += 1;
            continue;
        }
        if !matches!(merged[i].category, TokenCategory::Unknown) {
            i += 1;
            continue;
        }
        if !matches!(regex_list[i].category, TokenCategory::Unknown) {
            i += 1;
            continue;
        }
        let start = i;
        while i < n
            && !positions_list[i]
            && matches!(merged[i].category, TokenCategory::Unknown)
            && matches!(regex_list[i].category, TokenCategory::Unknown)
        {
            i += 1;
        }
        let lexeme: String = chars[start..i].iter().collect();
        if lexeme.is_empty() || lexeme.chars().all(|c| c.is_whitespace()) {
            continue;
        }
        out.push(LexicalIssue {
            lexeme,
            message: "unknown token".to_string(),
            char_index: Some(start),
        });
    }
    out
}

pub fn automata_regex_match(
    automata_list: &Vec<TokenStruct>,
    positions_list: &Vec<bool>,
    regex_list: &Vec<TokenStruct>,
    input: &str,
) -> (Vec<TokenStruct>, Vec<LexicalIssue>, Vec<TokenStruct>) {
    let unknown_token = TokenStruct {
        word: String::from("unknown"),
        rule: None,
        category: TokenCategory::Unknown,
    };

    let mut result = vec![unknown_token.clone(); automata_list.len()];
    let mut iterator = 0;

    for position in positions_list {
        let regex_has_longer_token = !matches!(regex_list[iterator].category, TokenCategory::Unknown)
            && regex_list[iterator].word.len() > automata_list[iterator].word.len();

        if *position && !regex_has_longer_token {
            let au = &automata_list[iterator];
            if matches!(au.category, TokenCategory::Unknown)
                && !matches!(regex_list[iterator].category, TokenCategory::Unknown)
            {
                result[iterator] = regex_list[iterator].clone();
            } else {
                result[iterator] = au.clone();
            }
        } else {
            result[iterator] = regex_list[iterator].clone();
        }
        iterator += 1;
    }

    let merge_issues = collect_unknown_runs(&result, input, positions_list, regex_list);
    let final_list = reclassify_reserved_identifiers(filter_unknowns(&result));
    (final_list, merge_issues, result)
}