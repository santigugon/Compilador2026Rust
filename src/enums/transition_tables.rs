use strum_macros::Display;

#[derive(Display,Copy,Clone)]
pub enum States{
    Q0,
    Q1,
    Q2,
    Q3,
    Q4,
    Q5_WHILE,
    Q6_OPEN_PAR,
    Q7_INT,
    Q8_FLOAT,
    Q9,
    Q10_IF,
    Q11,
    Q12,
    Q13,
    Q14_ELSE,
    Q15,
    Q16_ELIF,
    Q17_MINUS,
    Q18_PLUS,
    Q19_DEADSTATE,
    Q20_VAR,
    Q21_PROD,
    Q22_POW,
    Q23_DIV,
    Q24_FLOORDIV,
    Q25_MOD,
    Q26,
    Q27_STR
}

#[derive(Display)]
pub enum MachineStates{
    NONFINAL,
    FINALSTATE,
    DEADSTATE
}