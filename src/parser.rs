use crate::ast::AstNode;
use crate::enums::token_category::TokenCategory;
use crate::models::token_model::TokenStruct;

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
    pub position: Option<usize>,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.position {
            Some(position) => write!(f, "[ParseError at char {}] {}", position, self.message),
            None => write!(f, "[ParseError] {}", self.message),
        }
    }
}

pub struct Parser {
    tokens: Vec<TokenStruct>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<TokenStruct>) -> Self {
        let tokens = if tokens.is_empty() {
            vec![TokenStruct::recognized(
                "<EOF>".to_string(),
                TokenCategory::Unknown,
                0,
            )]
        } else {
            tokens
        };

        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> &TokenStruct {
        &self.tokens[self.pos]
    }

    fn peek_ahead(&self, k: usize) -> &TokenStruct {
        let idx = self.pos + k;
        if idx < self.tokens.len() {
            &self.tokens[idx]
        } else {
            &self.tokens[self.tokens.len() - 1]
        }
    }

    fn advance(&mut self) -> &TokenStruct {
        let tok = &self.tokens[self.pos];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn expect_word(&mut self, expected_word: &str) -> Result<(), ParseError> {
        let tok = self.peek();
        if tok.word == expected_word {
            self.advance();
            Ok(())
        } else {
            Err(ParseError {
                message: format!(
                    "Se esperaba '{}' pero llegó '{}' ({})",
                    expected_word, tok.word, tok.category
                ),
                position: tok.position,
            })
        }
    }

    fn expect_category(&mut self, expected: &TokenCategory) -> Result<String, ParseError> {
        let tok = self.peek();
        if &tok.category == expected {
            let word = tok.word.clone();
            self.advance();
            Ok(word)
        } else {
            Err(ParseError {
                message: format!(
                    "Se esperaba '{}' pero llegó '{}' ({})",
                    expected, tok.category, tok.word
                ),
                position: tok.position,
            })
        }
    }

    // program → decorator kernel
    // Retorna: AstNode::Program { kernel }
    pub fn parse_program(&mut self) -> Result<AstNode, ParseError> {
        self.parse_decorator()?;
        let kernel = self.parse_kernel()?;
        if self.pos < self.tokens.len() - 1 {
            let tok = self.peek();
            return Err(ParseError {
                message: format!(
                    "Se encontraron tokens extra después del único kernel permitido: '{}' ({})",
                    tok.word, tok.category
                ),
                position: tok.position,
            });
        }
        Ok(AstNode::Program {
            kernel: Box::new(kernel),
        })
    }

    // decorator → @ triton . jit
    // No produce nodo — solo valida que el decorador sea correcto
    fn parse_decorator(&mut self) -> Result<(), ParseError> {
        self.expect_word("@")?;
        self.expect_word("triton")?;
        self.expect_word(".")?;
        self.expect_word("jit")?;
        Ok(())
    }

    // kernel → def ID ( params ) : { stmt_list }
    // Retorna: AstNode::Kernel { name, params, body }
    fn parse_kernel(&mut self) -> Result<AstNode, ParseError> {
        self.expect_word("def")?;
        let name = self.expect_category(&TokenCategory::Identifier)?;
        self.expect_word("(")?;
        let params = self.parse_params()?;
        self.expect_word(")")?;
        self.expect_word(":")?;
        self.expect_word("{")?;
        let body = self.parse_stmt_list()?;
        self.expect_word("}")?;
        Ok(AstNode::Kernel { name, params, body })
    }

    // params → ε | param { , param }
    // Retorna: Vec<AstNode> — lista de Param
    fn parse_params(&mut self) -> Result<Vec<AstNode>, ParseError> {
        if self.peek().word == ")" {
            return Ok(vec![]);
        }
        let mut params = vec![self.parse_param()?];
        while self.peek().word == "," {
            self.advance();
            params.push(self.parse_param()?);
        }
        Ok(params)
    }

    // param → ID | ID : ID . ID
    // Retorna: AstNode::Param { name, annotation }
    fn parse_param(&mut self) -> Result<AstNode, ParseError> {
        let name = self.expect_category(&TokenCategory::Identifier)?;
        if self.peek().word == ":" {
            self.advance();
            self.expect_word("tl")?;
            self.expect_word(".")?;
            self.expect_word("constexpr")?;
            return Ok(AstNode::Param {
                name,
                annotation: Some("tl.constexpr".to_string()),
            });
        }
        Ok(AstNode::Param {
            name,
            annotation: None,
        })
    }

    // stmt_list → stmt { stmt }
    // Retorna: Vec<AstNode> — lista de sentencias
    fn parse_stmt_list(&mut self) -> Result<Vec<AstNode>, ParseError> {
        let mut stmts = vec![];
        while self.peek().word != "}" {
            stmts.push(self.parse_stmt()?);
        }
        Ok(stmts)
    }

    // stmt → assign | expr_stmt
    // LL(2): Identifier seguido de Assign → es asignación
    fn parse_stmt(&mut self) -> Result<AstNode, ParseError> {
        let is_assign = self.peek().category == TokenCategory::Identifier
            && self.peek_ahead(1).category == TokenCategory::Assign;

        if is_assign {
            self.parse_assign()
        } else {
            self.parse_expr_stmt()
        }
    }

    // assign → ID = expr ;
    // Retorna: AstNode::Assign { name, value }
    fn parse_assign(&mut self) -> Result<AstNode, ParseError> {
        let name = self.expect_category(&TokenCategory::Identifier)?;
        self.expect_category(&TokenCategory::Assign)?;
        let value = self.parse_expr()?;
        self.expect_word(";")?;
        Ok(AstNode::Assign {
            name,
            value: Box::new(value),
        })
    }

    // expr_stmt → expr ;
    // Retorna: AstNode::ExprStmt { expr }
    fn parse_expr_stmt(&mut self) -> Result<AstNode, ParseError> {
        let expr = self.parse_expr()?;
        self.expect_word(";")?;
        Ok(AstNode::ExprStmt {
            expr: Box::new(expr),
        })
    }

    // expr → term { (+ | -) term }
    // Retorna: AstNode::BinaryOp si hay operador, o el term directamente
    fn parse_expr(&mut self) -> Result<AstNode, ParseError> {
        let mut left = self.parse_term()?;
        while self.peek().word == "+" || self.peek().word == "-" {
            let op = self.advance().word.clone();
            let right = self.parse_term()?;
            left = AstNode::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    // term → factor { (* | /) factor }
    // Retorna: AstNode::BinaryOp si hay operador, o el factor directamente
    fn parse_term(&mut self) -> Result<AstNode, ParseError> {
        let mut left = self.parse_factor()?;
        while self.peek().word == "*" || self.peek().word == "/" {
            let op = self.advance().word.clone();
            let right = self.parse_factor()?;
            left = AstNode::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    // factor → ( expr ) | NUMBER | name_or_call
    // Retorna el nodo correspondiente a cada caso
    fn parse_factor(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.peek();

        if tok.word == "(" {
            self.advance();
            let node = self.parse_expr()?;
            self.expect_word(")")?;
            Ok(node)
        } else if tok.category == TokenCategory::Integer || tok.category == TokenCategory::Float {
            let value = self.advance().word.clone();
            Ok(AstNode::Number { value })
        } else if tok.category == TokenCategory::Identifier {
            self.parse_name_or_call()
        } else {
            Err(ParseError {
                message: format!(
                    "Se esperaba una expresión pero llegó '{}' ({})",
                    tok.word, tok.category
                ),
                position: tok.position,
            })
        }
    }

    // name_or_call → qualified_name [ ( args ) ]
    // Retorna: AstNode::Call si hay paréntesis, AstNode::Name si no
    fn parse_name_or_call(&mut self) -> Result<AstNode, ParseError> {
        let first = self.expect_category(&TokenCategory::Identifier)?;

        let callee = if self.peek().word == "." {
            self.advance();
            let second = self.expect_category(&TokenCategory::Identifier)?;
            format!("{}.{}", first, second)
        } else {
            first.clone()
        };

        if self.peek().word == "(" {
            self.advance();
            let args = self.parse_args()?;
            self.expect_word(")")?;
            Ok(AstNode::Call { callee, args })
        } else {
            Ok(AstNode::Name { id: callee })
        }
    }

    // args → ε | expr { , expr }
    // Retorna: Vec<AstNode> — lista de argumentos
    fn parse_args(&mut self) -> Result<Vec<AstNode>, ParseError> {
        if self.peek().word == ")" {
            return Ok(vec![]);
        }
        if self.peek().category == TokenCategory::Identifier
            && self.peek_ahead(1).category == TokenCategory::Assign
        {
            return Err(ParseError {
                message: format!(
                    "Argumentos nombrados no están soportados en Mini-Triton (cerca de '{}=')",
                    self.peek().word
                ),
                position: self.peek().position,
            });
        }
        let mut args = vec![self.parse_expr()?];
        while self.peek().word == "," {
            self.advance();
            args.push(self.parse_expr()?);
        }
        Ok(args)
    }
}

#[cfg(test)]
mod tests {
    use super::Parser;
    use crate::enums::token_category::TokenCategory;
    use crate::models::token_model::TokenStruct;

    fn token(word: &str, category: TokenCategory, position: usize) -> TokenStruct {
        TokenStruct::recognized(word.to_string(), category, position)
    }

    #[test]
    fn rejects_non_constexpr_annotation() {
        let tokens = vec![
            token("@", TokenCategory::Delimiter, 0),
            token("triton", TokenCategory::Identifier, 1),
            token(".", TokenCategory::Delimiter, 7),
            token("jit", TokenCategory::Identifier, 8),
            token("def", TokenCategory::Keyword, 12),
            token("typed", TokenCategory::Identifier, 16),
            token("(", TokenCategory::Delimiter, 21),
            token("x", TokenCategory::Identifier, 22),
            token(":", TokenCategory::Delimiter, 23),
            token("foo", TokenCategory::Identifier, 25),
            token(".", TokenCategory::Delimiter, 28),
            token("bar", TokenCategory::Identifier, 29),
            token(")", TokenCategory::Delimiter, 32),
            token(":", TokenCategory::Delimiter, 33),
            token("{", TokenCategory::Delimiter, 35),
            token("y", TokenCategory::Identifier, 39),
            token("=", TokenCategory::Assign, 41),
            token("x", TokenCategory::Identifier, 43),
            token(";", TokenCategory::Delimiter, 44),
            token("}", TokenCategory::Delimiter, 46),
        ];

        let error = Parser::new(tokens).parse_program().unwrap_err();
        assert!(error.message.contains("tl"));
        assert_eq!(error.position, Some(25));
    }

    #[test]
    fn empty_input_returns_parse_error_instead_of_panicking() {
        let error = Parser::new(vec![]).parse_program().unwrap_err();
        assert!(error.message.contains("@"));
    }
}
