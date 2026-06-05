use crate::enums::token_category::TokenCategory;
use crate::models::token_model::TokenStruct;

fn is_supported_arithmetic_operator(token: &TokenStruct) -> bool {
    matches!(token.word.as_str(), "+" | "-" | "*" | "/")
}

fn can_end_operand(token: &TokenStruct) -> bool {
    matches!(
        token.category,
        TokenCategory::Integer | TokenCategory::Float | TokenCategory::Identifier
    ) || token.word == ")"
}

fn can_start_operand(token: &TokenStruct) -> bool {
    matches!(
        token.category,
        TokenCategory::Integer | TokenCategory::Float | TokenCategory::Identifier
    ) || token.word == "("
}

pub fn validate_operator_operands(tokens: &[TokenStruct]) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    for (i, token) in tokens.iter().enumerate() {
        if !is_supported_arithmetic_operator(token) {
            continue;
        }

        let previous = i.checked_sub(1).and_then(|idx| tokens.get(idx));
        let next = tokens.get(i + 1);

        if previous.is_none_or(|tok| !can_end_operand(tok)) {
            errors.push(format!(
                "operator '{}' at token index {}: must be preceded by an identifier, number, or ')'",
                token.word, i
            ));
        }

        if next.is_none_or(|tok| !can_start_operand(tok)) {
            errors.push(format!(
                "operator '{}' at token index {}: must be followed by an identifier, number, or '('",
                token.word, i
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::validate_operator_operands;
    use crate::enums::token_category::TokenCategory;
    use crate::models::token_model::TokenStruct;

    fn token(word: &str, category: TokenCategory, position: usize) -> TokenStruct {
        TokenStruct::recognized(word.to_string(), category, position)
    }

    #[test]
    fn accepts_operator_after_parenthesized_expression() {
        let tokens = vec![
            token("(", TokenCategory::Delimiter, 0),
            token("x", TokenCategory::Identifier, 1),
            token("+", TokenCategory::Operator, 3),
            token("1", TokenCategory::Integer, 5),
            token(")", TokenCategory::Delimiter, 6),
            token("*", TokenCategory::Operator, 8),
            token("2", TokenCategory::Integer, 10),
        ];

        assert!(validate_operator_operands(&tokens).is_ok());
    }

    #[test]
    fn accepts_operator_after_function_call() {
        let tokens = vec![
            token("tl", TokenCategory::Identifier, 0),
            token(".", TokenCategory::Delimiter, 2),
            token("load", TokenCategory::Identifier, 3),
            token("(", TokenCategory::Delimiter, 7),
            token("x", TokenCategory::Identifier, 8),
            token(")", TokenCategory::Delimiter, 9),
            token("+", TokenCategory::Operator, 11),
            token("1", TokenCategory::Integer, 13),
        ];

        assert!(validate_operator_operands(&tokens).is_ok());
    }
}
