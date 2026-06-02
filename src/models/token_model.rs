use regex::Regex;

use crate::enums::token_category::TokenCategory;

#[derive(Clone)]
pub struct TokenStruct {
    pub word: String,
    pub rule: Option<Regex>,
    pub category: TokenCategory,
    pub position: Option<usize>,
}

impl TokenStruct {
    pub fn pattern(word: &str, rule: Regex, category: TokenCategory) -> Self {
        Self {
            word: word.to_string(),
            rule: Some(rule),
            category,
            position: None,
        }
    }

    pub fn recognized(word: String, category: TokenCategory, position: usize) -> Self {
        Self {
            word,
            rule: None,
            category,
            position: Some(position),
        }
    }

    pub fn unknown() -> Self {
        Self {
            word: String::from("unknown"),
            rule: None,
            category: TokenCategory::Unknown,
            position: None,
        }
    }
}
