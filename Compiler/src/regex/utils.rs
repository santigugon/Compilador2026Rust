use regex::Regex;

use crate::enums::token_category::TokenCategory;
use crate::models::token_model::TokenStruct;

fn regex_list() -> Vec<TokenStruct> {
    return vec![
        TokenStruct::pattern(
            "def",
            Regex::new(r"\bdef\b").unwrap(),
            TokenCategory::Keyword,
        ),
        TokenStruct::pattern(
            "return",
            Regex::new(r"\breturn\b").unwrap(),
            TokenCategory::Keyword,
        ),
        TokenStruct::pattern(
            "for",
            Regex::new(r"\bfor\b").unwrap(),
            TokenCategory::Keyword,
        ),
        TokenStruct::pattern("in", Regex::new(r"\bin\b").unwrap(), TokenCategory::Keyword),
        TokenStruct::pattern("is", Regex::new(r"\bis\b").unwrap(), TokenCategory::Keyword),
        TokenStruct::pattern(
            "and",
            Regex::new(r"\band\b").unwrap(),
            TokenCategory::Keyword,
        ),
        TokenStruct::pattern("or", Regex::new(r"\bor\b").unwrap(), TokenCategory::Keyword),
        TokenStruct::pattern(
            "not",
            Regex::new(r"\bnot\b").unwrap(),
            TokenCategory::Keyword,
        ),
        TokenStruct::pattern(
            "True",
            Regex::new(r"\bTrue\b").unwrap(),
            TokenCategory::Keyword,
        ),
        TokenStruct::pattern(
            "False",
            Regex::new(r"\bFalse\b").unwrap(),
            TokenCategory::Keyword,
        ),
        TokenStruct::pattern(
            "None",
            Regex::new(r"\bNone\b").unwrap(),
            TokenCategory::Keyword,
        ),
        TokenStruct::pattern(
            "pass",
            Regex::new(r"\bpass\b").unwrap(),
            TokenCategory::Keyword,
        ),
        TokenStruct::pattern(
            "break",
            Regex::new(r"\bbreak\b").unwrap(),
            TokenCategory::Keyword,
        ),
        TokenStruct::pattern(
            "continue",
            Regex::new(r"\bcontinue\b").unwrap(),
            TokenCategory::Keyword,
        ),
        TokenStruct::pattern(
            "float",
            Regex::new(r"\b[0-9]+\.[0-9]+\b").unwrap(),
            TokenCategory::Float,
        ),
        TokenStruct::pattern(
            "int",
            Regex::new(r"\b[0-9]+\b").unwrap(),
            TokenCategory::Integer,
        ),
        TokenStruct::pattern("<=", Regex::new(r"<=").unwrap(), TokenCategory::Operator),
        TokenStruct::pattern(">=", Regex::new(r">=").unwrap(), TokenCategory::Operator),
        TokenStruct::pattern("==", Regex::new(r"==").unwrap(), TokenCategory::Operator),
        TokenStruct::pattern("!=", Regex::new(r"!=").unwrap(), TokenCategory::Operator),
        TokenStruct::pattern("+=", Regex::new(r"\+=").unwrap(), TokenCategory::Operator),
        TokenStruct::pattern("-=", Regex::new(r"-=").unwrap(), TokenCategory::Operator),
        TokenStruct::pattern("*=", Regex::new(r"\*=").unwrap(), TokenCategory::Operator),
        TokenStruct::pattern("/=", Regex::new(r"/=").unwrap(), TokenCategory::Operator),
        TokenStruct::pattern("->", Regex::new(r"->").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern("<<", Regex::new(r"<<").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern(">>", Regex::new(r">>").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern("<", Regex::new(r"<").unwrap(), TokenCategory::Operator),
        TokenStruct::pattern(">", Regex::new(r">").unwrap(), TokenCategory::Operator),
        TokenStruct::pattern("=", Regex::new(r"=").unwrap(), TokenCategory::Assign),
        TokenStruct::pattern("(", Regex::new(r"\(").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern(")", Regex::new(r"\)").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern("[", Regex::new(r"\[").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern("]", Regex::new(r"\]").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern("{", Regex::new(r"\{").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern("}", Regex::new(r"\}").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern(",", Regex::new(r",").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern(":", Regex::new(r":").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern(";", Regex::new(r";").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern(".", Regex::new(r"\.").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern("@", Regex::new(r"@").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern("~", Regex::new(r"~").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern("&", Regex::new(r"&").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern("|", Regex::new(r"\|").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern("^", Regex::new(r"\^").unwrap(), TokenCategory::Delimiter),
        TokenStruct::pattern(
            "id",
            Regex::new(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b").unwrap(),
            TokenCategory::Identifier,
        ),
    ];
}

pub fn regex_match(input: String) -> Vec<TokenStruct> {
    let unknown_token = TokenStruct::unknown();
    let regexes = regex_list();
    let mut token_references = vec![unknown_token; input.len()];
    let mut occupied_positions = vec![false; input.len()];

    for rule in regexes {
        match rule.rule {
            Some(ref regex) => {
                for m in regex.find_iter(&input) {
                    if occupied_positions[m.start()..m.end()].iter().any(|o| *o) {
                        continue;
                    }
                    let matched_text = m.as_str().to_string();

                    for position in m.start()..m.end() {
                        token_references[position] = TokenStruct::recognized(
                            matched_text.clone(),
                            rule.category.clone(),
                            m.start(),
                        );
                        occupied_positions[position] = true;
                    }
                }
            }
            None => {
                println!("No regex rule for {}", rule.word);
            }
        }
    }

    token_references
}
