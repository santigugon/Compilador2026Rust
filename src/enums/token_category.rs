use strum_macros::Display;

#[derive(Display, Clone)]
pub enum TokenCategory{
    Keyword,
    Identifier,
    Integer,
    Float,
    Operator,
    Delimiter,
    Indentation,
    WhiteSpace,
    StringToken,
    Unknown
}