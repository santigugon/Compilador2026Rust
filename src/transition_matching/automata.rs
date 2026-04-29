use crate::enums::transition_tables::States;
use crate::enums::transition_tables::MachineStates;
use crate::enums::token_category::TokenCategory;
use crate::models::token_model::TokenStruct;

pub fn state_match(c:char, current_state:States)->States{
    match (c,current_state){
        // WHILE
        ('w',States::Q0)=>States::Q1,
        ('h',States::Q1)=>States::Q2,
        ('i',States::Q2)=>States::Q3,
        ('l',States::Q3)=>States::Q4,
        ('e',States::Q4)=>States::Q5_WHILE,
        // IF,ELSE ELIF
        ('i',States::Q0)=>States::Q9,
        ('f',States::Q9)=>States::Q10_IF,
        ('e',States::Q0)=>States::Q11,
        ('l',States::Q11)=>States::Q12,
        ('s',States::Q12)=>States::Q13,
        ('e',States::Q13)=>States::Q14_ELSE,
        ('i',States::Q12)=>States::Q15,
        ('f',States::Q15)=>States::Q16_ELIF,
        // INT, FLOAT
        ('0'..='9', States::Q0) => States::Q7_INT,
        ('.', States::Q7_INT) => States::Q8_FLOAT,
        ('0'..='9', States::Q8_FLOAT) => States::Q8_FLOAT,
        // OPERATORS
        ('-',States::Q0)=>States::Q17_MINUS,
        ('+',States::Q0)=>States::Q18_PLUS,
        ('*',States::Q0)=>States::Q21_PROD,
        ('*',States::Q21_PROD)=>States::Q22_POW,
        ('/',States::Q0)=>States::Q23_DIV,
        ('/',States::Q23_DIV)=>States::Q24_FLOORDIV,
        ('%',States::Q0)=>States::Q25_MOD,

        // STRING
        ('"',States::Q0)=>States::Q26,
        ('\n',States::Q26)=>States::Q19_DEADSTATE,
        ('"',States::Q26)=>States::Q27_STR,
        (_,States::Q26)=>States::Q26,

        // VAR
        (c, States::Q0) if c.is_alphabetic() || c == '_' => States::Q20_VAR,
        (c, States::Q20_VAR) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,

        // Transition if, else, elif to potential var
        (c, States::Q1) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,
        (c, States::Q2) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,
        (c, States::Q3) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,
        (c, States::Q4) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,
        (c, States::Q9) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,
        (c, States::Q11) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,
        (c, States::Q12) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,
        (c, States::Q13) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,
        (c, States::Q15) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,
        (c, States::Q5_WHILE) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,
        (c, States::Q10_IF) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,
        (c, States::Q14_ELSE) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,
        (c, States::Q16_ELIF) if c.is_alphanumeric() || c == '_' => States::Q20_VAR,

        // Any transition with white space or new line resets the state to 0
        (' ', _)=>States::Q0,
        ('\n', _)=>States::Q0,

        // Any non explicit transition goes to death state
        (_,_)=>States::Q19_DEADSTATE
    }
}

pub fn transform_to_machine_state(state:States) ->MachineStates{
    match(state){
        States::Q5_WHILE=>MachineStates::FINALSTATE,
        States::Q6_OPEN_PAR=>MachineStates::FINALSTATE,
        States::Q7_INT=>MachineStates::FINALSTATE,
        States::Q8_FLOAT=>MachineStates::FINALSTATE,
        States::Q14_ELSE=>MachineStates::FINALSTATE,
        States::Q16_ELIF=>MachineStates::FINALSTATE,
        States::Q17_MINUS=>MachineStates::FINALSTATE,
        States::Q18_PLUS=>MachineStates::FINALSTATE,
        States::Q20_VAR=>MachineStates::FINALSTATE,
        States::Q21_PROD=>MachineStates::FINALSTATE,
        States::Q22_POW=>MachineStates::FINALSTATE,
        States::Q23_DIV=>MachineStates::FINALSTATE,
        States::Q24_FLOORDIV=>MachineStates::FINALSTATE,
        States::Q25_MOD=>MachineStates::FINALSTATE,
        States::Q27_STR=>MachineStates::FINALSTATE,
        States::Q19_DEADSTATE=>MachineStates::DEADSTATE,
        _=>MachineStates::NONFINAL
    }
}

pub fn lookahead_makes_final_state_valid(next_c:char,current_state:States)->MachineStates{
    match(next_c, current_state){
        ('(', States::Q5_WHILE)=>MachineStates::FINALSTATE,
        ('(', States::Q10_IF)=>MachineStates::FINALSTATE,
        ('(', States::Q16_ELIF)=>MachineStates::FINALSTATE,

        ('*', States::Q21_PROD)=>MachineStates::NONFINAL,
        ('/', States::Q23_DIV)=>MachineStates::NONFINAL,

        (_, States::Q21_PROD)=>MachineStates::FINALSTATE,
        (_, States::Q23_DIV)=>MachineStates::FINALSTATE,

        (_,_)=>MachineStates::NONFINAL
    }
}

pub fn requires_lookeahead(current_state:States)->bool{
    match(current_state){
        States::Q5_WHILE=>true,
        States::Q10_IF=>true,
        States::Q16_ELIF=>true,

        States::Q21_PROD=>true,
        States::Q23_DIV=>true,
        _=>false
    }
}

pub fn is_operator(current_state:States)->bool{
    match(current_state){
        States::Q17_MINUS=>true,
        States::Q18_PLUS=>true,
        States::Q21_PROD=>true,
        States::Q22_POW=>true,
        States::Q23_DIV=>true,
        States::Q24_FLOORDIV=>true,
        States::Q25_MOD=>true,
        _=>false
    }
}

pub fn match_transitions(input:String){
    let mut current_state=States::Q0;

    let unknown_token= TokenStruct{
        word:String::from("unknown"),
        rule:None,
        category: TokenCategory::Unknown
    };

    let mut token_references= vec![unknown_token;input.len()];

    let mut first_iteration= true;
    let mut past_char= 'a';
    let mut iterator=0;

    let mut token_word = String::new();

    for ch in input.chars() {
        let next_char= ch;

        if(first_iteration){
            past_char= next_char;
            first_iteration=false;
            break;
        }
        
        let lookahead = requires_lookeahead(current_state);
        let machine_state_enum= transform_to_machine_state(current_state);
        
        current_state=state_match(past_char, current_state);
        
        let token_finalization:bool= !lookahead && (next_char==' '|| next_char == '\n' || is_operator(current_state));

        
        
        
        past_char = next_char;
        
        if( past_char != ' '){
            token_word.push(past_char);
        }


        match machine_state_enum {
            MachineStates::FINALSTATE if(token_finalization)=> {
                let new_token =TokenStruct{
                    word:token_word,
                    rule:None,
                    category: TokenCategory::Unknown
                };
                token_references[iterator]= new_token;
                token_word = String::new();
            },
            _=>{}
        }
        

        iterator+=1;
    }
}