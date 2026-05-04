use crate::models::token_model::TokenStruct;
use crate::enums::token_category::TokenCategory;


pub fn filter_unknowns(final_list: &Vec<TokenStruct>)->Vec<TokenStruct>{
    let unknown_token= TokenStruct{
            word:String::from("unknown"),
            rule:None,
            category: TokenCategory::Unknown
    };

    let mut result: Vec<TokenStruct> = vec![];

    for token in final_list{
        match (token.category){
            TokenCategory::Unknown=>{
                continue;
            }
            TokenCategory::Indentation=>{
                continue;
            }
            TokenCategory::WhiteSpace=>{
                continue;
            }
            _=>{
                result.push(token.clone());
            }
        }
    }

    return result;
}

pub fn automata_regex_match(automata_list:&Vec<TokenStruct>, positions_list:&Vec<bool> ,regex_list:&Vec<TokenStruct>)->Vec<TokenStruct>{

    let unknown_token= TokenStruct{
        word:String::from("unknown"),
        rule:None,
        category: TokenCategory::Unknown
    };

    let mut result= vec![unknown_token.clone();automata_list.len()];
    let mut iterator=0;

    for position in positions_list{
        if *position{
            result[iterator]=automata_list[iterator].clone();
        }
        else{
            result[iterator]=regex_list[iterator].clone();
        }
        iterator+=1;
    }

    return filter_unknowns(&result);
}