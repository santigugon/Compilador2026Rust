#[derive(Clone)]
pub enum TokenCategory{
    Keyword,
    Identifier,
    Operator,
    Delimiter,
    Identation,
    WhiteSpace,
    Unknown
}