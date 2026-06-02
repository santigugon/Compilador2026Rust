use crate::enums::token_category::TokenCategory;
use crate::models::token_model::TokenStruct;
use crate::ast::AstNode;

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[ParseError] {}", self.message)
    }
}

pub struct Parser {
    tokens: Vec<TokenStruct>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<TokenStruct>) -> Self {
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
            let ns = self.expect_category(&TokenCategory::Identifier)?;  // tl
            self.expect_word(".")?;
            let ann = self.expect_category(&TokenCategory::Identifier)?; // constexpr
            return Ok(AstNode::Param {
                name,
                annotation: Some(format!("{}.{}", ns, ann)),
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
        } else if tok.category == TokenCategory::Integer
            || tok.category == TokenCategory::Float
        {
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
