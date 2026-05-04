use crate::enums::transition_tables::States;
use crate::enums::transition_tables::MachineStates;
use crate::enums::token_category::TokenCategory;
use crate::models::token_model::TokenStruct;

fn state_match(c:char, current_state:States)->States{
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
        // INT
        ('0'..='9', States::Q0) => States::Q7_INT,
        ('0'..='9', States::Q7_INT) => States::Q7_INT,

        // FLOAT
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

fn transform_to_machine_state(state:States) ->MachineStates{
    match(state){
        States::Q5_WHILE=>MachineStates::FINALSTATE,
        States::Q6_OPEN_PAR=>MachineStates::FINALSTATE,
        States::Q7_INT=>MachineStates::FINALSTATE,
        States::Q8_FLOAT=>MachineStates::FINALSTATE,
        States::Q10_IF=>MachineStates::FINALSTATE,
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

fn lookahead_makes_final_state_valid(next_c:char,current_state:States)->bool{
    match(next_c, current_state){
        ('(', States::Q5_WHILE)=>true,
        ('(', States::Q10_IF)=>true,
        ('(', States::Q16_ELIF)=>true,

        ('*', States::Q21_PROD)=>false,
        ('/', States::Q23_DIV)=>false,

        (_, States::Q21_PROD)=>true,
        (_, States::Q23_DIV)=>true,

        (' ',_)=>true,

        (_,_)=>false
    }
}

fn requires_lookeahead(current_state:States)->bool{
    match(current_state){
        States::Q5_WHILE=>true,
        States::Q10_IF=>true,
        States::Q16_ELIF=>true,

        States::Q21_PROD=>true,
        States::Q23_DIV=>true,
        _=>false
    }
}

fn is_operator(current_state:States)->bool{
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

fn state_category_matching(state:States)->TokenCategory{
    match(state) {
        States::Q5_WHILE=>TokenCategory::Keyword,
        States::Q6_OPEN_PAR=>TokenCategory::Delimiter,
        States::Q7_INT=>TokenCategory::Integer,
        States::Q8_FLOAT=>TokenCategory::Float,
        States::Q10_IF=>TokenCategory::Keyword,
        States::Q14_ELSE=>TokenCategory::Keyword,
        States::Q16_ELIF=>TokenCategory::Keyword,
        States::Q17_MINUS=>TokenCategory::Operator,
        States::Q18_PLUS=>TokenCategory::Operator,
        States::Q20_VAR=>TokenCategory::Identifier,
        States::Q21_PROD=>TokenCategory::Operator,
        States::Q22_POW=>TokenCategory::Operator,
        States::Q23_DIV=>TokenCategory::Operator,
        States::Q24_FLOORDIV=>TokenCategory::Operator,
        States::Q25_MOD=>TokenCategory::Operator,
        States::Q27_STR=>TokenCategory::StringToken,
        States::Q19_DEADSTATE=>TokenCategory::Unknown,

        _=>TokenCategory::Unknown
    }
}

pub fn return_used_positions(tokens: &Vec<TokenStruct>)->Vec<bool>{
    let mut result =vec![false; tokens.len()];
    let last_ind =tokens.len()-1;
    let mut fill_with_true_till=last_ind;

    for i in 0..=last_ind{
        let iterator= last_ind-i;
        let token = &tokens[iterator];

        if (fill_with_true_till<=iterator){
            result[iterator]=true;
            continue;
        }

        match(token.category){
            TokenCategory::Unknown=>{
                continue;
            }
            _=>{
                let word_len = token.word.len();

                fill_with_true_till = iterator.saturating_sub(word_len - 1);
                result[iterator]=true;
            }
        }
    }
    return result;

}

pub fn match_transitions(input:&String)->Vec<TokenStruct>{

    println!("HEllooo {}", input);
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
            continue;
        }

        // println!("Current state {}", current_state.clone());
        // println!("Char and next char {}-{} ",past_char.clone(),next_char.clone());
        current_state=state_match(past_char, current_state);
        // println!("New state {} \n", current_state.clone());
        
        let lookahead = requires_lookeahead(current_state);
        let lookahead_is_final =lookahead_makes_final_state_valid(next_char.clone(),current_state);
        
        let token_finalization:bool= (!lookahead||lookahead_is_final) && (next_char==' '|| next_char == '\n' || is_operator(current_state));
        // println!("\n Lookahead is final {} \n", lookahead_is_final);
        // println!("Token finalizatio {} \n", token_finalization);
        
        let machine_state_enum= transform_to_machine_state(current_state);
        
        if( past_char != ' ' && past_char != '\n'){
            println!("the past char {}", past_char);
            token_word.push(past_char);
        }
        
        let current_category = state_category_matching(current_state);
        
        match machine_state_enum {
            MachineStates::FINALSTATE if(token_finalization)=> {
                let new_token =TokenStruct{
                    word:token_word.clone(),
                    rule:None,
                    category: current_category.clone()
                };
                // println!("\n Iterator {} Word: {} Category {} \n", iterator, token_word, current_category);
                token_references[iterator]= new_token;
                token_word = String::new();
                current_state = States::Q0;
            },
            MachineStates::DEADSTATE=>{
                token_word=String::new();
            } 
            _=>{}
        }
        
        past_char = next_char;
        iterator+=1;
    }

    return token_references;
}