use regex::Regex;

use crate::models::token_model::TokenStruct;
use crate::enums::token_category::TokenCategory;

pub fn regex_match(){
    let re = Regex::new(r"\bdef\b").unwrap();
    let test= "def Hola como estas def def defaaa jajaijdisajd desf";

    
    let unknown_token= TokenStruct{
        word:String::from("unknown"),
        rule:Some(Regex::new(r"$^").unwrap()),
        category: TokenCategory::Unknown
    };
    
    let def_ex= TokenStruct{
        word:String::from("def"),
        rule:Some(Regex::new(r"\bdef\b").unwrap()),
        category: TokenCategory::Keyword
    };

    let regexes:[&TokenStruct;2]= [&unknown_token, &def_ex];
    let mut token_references= vec![&unknown_token;test.len()];
    let mut counter=1;

    // let results: Vec<&str> = re.find_iter(test).map(|m| m.as_str()).collect();

    
    for rule in regexes {
        println!("Rule, {}", rule.word);

        match &rule.rule {
            Some(regex) => {
                for m in regex.find_iter(test) {
                    println!("Hello the result is, {}:{}", m.start(), m.end());
                    token_references[m.start()] = rule;
                    println!("count {}", counter);
                    counter += 1;
                }
            }
            None => {
                println!("No regex rule for {}", rule.word);
            }
        }
    }
    
    for token in &token_references{
        println!("Token,{ }", token.word);
    }
    // println!("Array size,{}", token_references.len());

}