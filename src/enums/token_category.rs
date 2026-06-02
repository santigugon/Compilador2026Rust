use strum_macros::Display;

#[derive(Display, Clone, PartialEq, Eq)]
pub enum TokenCategory {
    Keyword,
    Identifier,
    Integer,
    Float,
    Operator,
    Assign,
    Delimiter,
    Indentation,
    WhiteSpace,
    StringToken,
    Unknown,
}
