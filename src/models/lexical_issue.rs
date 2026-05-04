#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LexicalIssue {
    pub lexeme: String,
    pub message: String,
    /// Character index in the source line/file where this issue was detected (ASCII-aligned).
    pub char_index: Option<usize>,
}
