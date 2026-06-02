#[derive(Debug)]
pub enum AstNode {
    Program {
        kernel: Box<AstNode>,
    },

    Kernel {
        name: String,
        params: Vec<AstNode>,
        body: Vec<AstNode>,
    },

    Param {
        name: String,
        annotation: Option<String>,
    },

    Assign {
        name: String,
        value: Box<AstNode>,
    },

    ExprStmt {
        expr: Box<AstNode>,
    },

    BinaryOp {
        op: String,
        left: Box<AstNode>,
        right: Box<AstNode>,
    },

    Call {
        callee: String,
        args: Vec<AstNode>,
    },

    Name {
        id: String,
    },

    Number {
        value: String,
    },
}

impl AstNode {
    pub fn pretty(&self, indent: usize) -> String {
        let pad = "  ".repeat(indent);
        match self {
            AstNode::Program { kernel } => {
                format!("Program(\n{}\n)", kernel.pretty(indent + 1))
            }
            AstNode::Kernel { name, params, body } => {
                let params_str = params
                    .iter()
                    .map(|p| p.pretty(0))
                    .collect::<Vec<_>>()
                    .join(", ");
                let body_str = body
                    .iter()
                    .map(|s| format!("{}{}", "  ".repeat(indent + 2), s.pretty(indent + 2)))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "{}Kernel(name=\"{}\", params=[{}], body=[\n{}\n{}])",
                    pad, name, params_str, body_str, pad
                )
            }
            AstNode::Param { name, annotation } => match annotation {
                Some(ann) => format!("{}:{}", name, ann),
                None => name.clone(),
            },
            AstNode::Assign { name, value } => {
                format!("{}Assign({}, {})", pad, name, value.pretty(0))
            }
            AstNode::ExprStmt { expr } => {
                format!("{}ExprStmt({})", pad, expr.pretty(0))
            }
            AstNode::BinaryOp { op, left, right } => {
                format!("BinaryOp(\"{}\", {}, {})", op, left.pretty(0), right.pretty(0))
            }
            AstNode::Call { callee, args } => {
                let args_str = args
                    .iter()
                    .map(|a| a.pretty(0))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("Call(\"{}\", [{}])", callee, args_str)
            }
            AstNode::Name { id } => format!("Name({})", id),
            AstNode::Number { value } => format!("Number({})", value),
        }
    }
}