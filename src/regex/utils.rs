use regex::Regex;

use crate::models::regex_rule::RegexRule;
use crate::enums::token_category::TokenCategory;

pub fn regex_match(){
    let re = Regex::new(r"\bdef\b").unwrap();
    let test= "def Hola como estas def def defaaa jajaijdisajd desf";

    
    let ex_undef= RegexRule{
        word:String::from("undefined"),
        rule:Regex::new(r"$^").unwrap(),
        category: TokenCategory::Keyword
    };
    
    let def_ex= RegexRule{
        word:String::from("def"),
        rule:Regex::new(r"\bdef\b").unwrap(),
        category: TokenCategory::Keyword
    };

    let regexes:[&RegexRule;2]= [&ex_undef, &def_ex];
    let mut token_references= vec![&ex_undef;test.len()];
    let mut counter=1;

    // let results: Vec<&str> = re.find_iter(test).map(|m| m.as_str()).collect();

    
    for rule in regexes{
        println!("Rule,{}", rule.word);
        for m in rule.rule.find_iter(test){
            println!("Hello the result is, {}:{}", m.start(), m.end());
            token_references[m.start()]=&rule;
            println!("count {}", counter);
            counter=counter+1;
        }
    }
    
    for token in &token_references{
        println!("Token,{ }", token.word);
    }
    // println!("Array size,{}", token_references.len());

}