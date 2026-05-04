use crate::enums::token_category::TokenCategory;
use crate::models::token_model::TokenStruct;

enum SemanticState {
    ExpectOperand {
        pending_operator: Option<(usize, String)>,
    },
    ExpectOperator,
}

fn is_operand(cat: &TokenCategory) -> bool {
    matches!(
        cat,
        TokenCategory::Integer | TokenCategory::Float | TokenCategory::Identifier
    )
}

fn is_expression_boundary(token: &TokenStruct) -> bool {
    matches!(&token.category, TokenCategory::Delimiter | TokenCategory::Keyword)
}

fn operator_missing_right_operand_error(index: usize, word: &str) -> String {
    format!(
        "operator '{}' at token index {}: must be followed by an integer, float, or identifier",
        word, index
    )
}

pub fn validate_operator_operands(tokens: &[TokenStruct]) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let mut state = SemanticState::ExpectOperand {
        pending_operator: None,
    };

    for (i, token) in tokens.iter().enumerate() {
        match (&state, &token.category) {
            (_, category) if is_operand(category) => {
                state = SemanticState::ExpectOperator;
            }
            (SemanticState::ExpectOperator, TokenCategory::Operator) => {
                state = SemanticState::ExpectOperand {
                    pending_operator: Some((i, token.word.clone())),
                };
            }
            (SemanticState::ExpectOperand { pending_operator }, TokenCategory::Operator) => {
                if let Some((operator_index, operator_word)) = pending_operator {
                    errors.push(operator_missing_right_operand_error(
                        *operator_index,
                        operator_word,
                    ));
                }
                errors.push(format!(
                    "operator '{}' at token index {}: must be preceded by an integer, float, or identifier",
                    token.word, i
                ));
                state = SemanticState::ExpectOperand {
                    pending_operator: Some((i, token.word.clone())),
                };
            }
            (SemanticState::ExpectOperand { pending_operator }, _) if is_expression_boundary(token) => {
                if let Some((operator_index, operator_word)) = pending_operator {
                    errors.push(operator_missing_right_operand_error(
                        *operator_index,
                        operator_word,
                    ));
                }
                state = SemanticState::ExpectOperand {
                    pending_operator: None,
                };
            }
            (SemanticState::ExpectOperator, _) if is_expression_boundary(token) => {
                state = SemanticState::ExpectOperand {
                    pending_operator: None,
                };
            }
            _ => {
                state = SemanticState::ExpectOperand {
                    pending_operator: None,
                };
            }
        }
    }

    if let SemanticState::ExpectOperand {
        pending_operator: Some((operator_index, operator_word)),
    } = state
    {
        errors.push(operator_missing_right_operand_error(
            operator_index,
            &operator_word,
        ));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
