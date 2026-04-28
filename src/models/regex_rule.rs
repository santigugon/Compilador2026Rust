use regex::Regex;

use crate::enums::token_category::TokenCategory;

pub struct RegexRule{
    pub word:String,
    pub rule: Regex,
    pub category: TokenCategory
}