use regex::Regex;

use crate::enums::token_category::TokenCategory;

#[derive(Clone)]
pub struct TokenStruct{
    pub word:String,
    pub rule: Option<Regex>,
    pub category: TokenCategory
}