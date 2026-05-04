#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LexicalIssue {
    pub lexeme: String,
    pub message: String,
    pub char_index: Option<usize>,
}
