use regex::Regex;

use crate::models::token_model::TokenStruct;
use crate::enums::token_category::TokenCategory;


fn regex_list() -> Vec<TokenStruct> {
    return vec![
        TokenStruct { word: String::from("def"), rule: Some(Regex::new(r"\bdef\b").unwrap()), category: TokenCategory::Keyword },
        TokenStruct { word: String::from("return"), rule: Some(Regex::new(r"\breturn\b").unwrap()), category: TokenCategory::Keyword },
        TokenStruct { word: String::from("for"), rule: Some(Regex::new(r"\bfor\b").unwrap()), category: TokenCategory::Keyword },
        TokenStruct { word: String::from("in"), rule: Some(Regex::new(r"\bin\b").unwrap()), category: TokenCategory::Keyword },
        TokenStruct { word: String::from("is"), rule: Some(Regex::new(r"\bis\b").unwrap()), category: TokenCategory::Keyword },
        TokenStruct { word: String::from("and"), rule: Some(Regex::new(r"\band\b").unwrap()), category: TokenCategory::Keyword },
        TokenStruct { word: String::from("or"), rule: Some(Regex::new(r"\bor\b").unwrap()), category: TokenCategory::Keyword },
        TokenStruct { word: String::from("not"), rule: Some(Regex::new(r"\bnot\b").unwrap()), category: TokenCategory::Keyword },
        TokenStruct { word: String::from("True"), rule: Some(Regex::new(r"\bTrue\b").unwrap()), category: TokenCategory::Keyword },
        TokenStruct { word: String::from("False"), rule: Some(Regex::new(r"\bFalse\b").unwrap()), category: TokenCategory::Keyword },
        TokenStruct { word: String::from("None"), rule: Some(Regex::new(r"\bNone\b").unwrap()), category: TokenCategory::Keyword },
        TokenStruct { word: String::from("pass"), rule: Some(Regex::new(r"\bpass\b").unwrap()), category: TokenCategory::Keyword },
        TokenStruct { word: String::from("break"), rule: Some(Regex::new(r"\bbreak\b").unwrap()), category: TokenCategory::Keyword },
        TokenStruct { word: String::from("continue"), rule: Some(Regex::new(r"\bcontinue\b").unwrap()), category: TokenCategory::Keyword },

        TokenStruct { word: String::from("<="), rule: Some(Regex::new(r"<=").unwrap()), category: TokenCategory::Operator },
        TokenStruct { word: String::from(">="), rule: Some(Regex::new(r">=").unwrap()), category: TokenCategory::Operator },
        TokenStruct { word: String::from("=="), rule: Some(Regex::new(r"==").unwrap()), category: TokenCategory::Operator },
        TokenStruct { word: String::from("!="), rule: Some(Regex::new(r"!=").unwrap()), category: TokenCategory::Operator },
        TokenStruct { word: String::from("+="), rule: Some(Regex::new(r"\+=").unwrap()), category: TokenCategory::Operator },
        TokenStruct { word: String::from("-="), rule: Some(Regex::new(r"-=").unwrap()), category: TokenCategory::Operator },
        TokenStruct { word: String::from("*="), rule: Some(Regex::new(r"\*=").unwrap()), category: TokenCategory::Operator },
        TokenStruct { word: String::from("/="), rule: Some(Regex::new(r"/=").unwrap()), category: TokenCategory::Operator },
        TokenStruct { word: String::from("<"), rule: Some(Regex::new(r"<").unwrap()), category: TokenCategory::Operator },
        TokenStruct { word: String::from(">"), rule: Some(Regex::new(r">").unwrap()), category: TokenCategory::Operator },
        TokenStruct { word: String::from("="), rule: Some(Regex::new(r"=").unwrap()), category: TokenCategory::Operator },

        TokenStruct { word: String::from("->"), rule: Some(Regex::new(r"->").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from("<<"), rule: Some(Regex::new(r"<<").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from(">>"), rule: Some(Regex::new(r">>").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from("("), rule: Some(Regex::new(r"\(").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from(")"), rule: Some(Regex::new(r"\)").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from("["), rule: Some(Regex::new(r"\[").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from("]"), rule: Some(Regex::new(r"\]").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from("{"), rule: Some(Regex::new(r"\{").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from("}"), rule: Some(Regex::new(r"\}").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from(","), rule: Some(Regex::new(r",").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from(":"), rule: Some(Regex::new(r":").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from("."), rule: Some(Regex::new(r"\.").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from("@"), rule: Some(Regex::new(r"@").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from("~"), rule: Some(Regex::new(r"~").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from("&"), rule: Some(Regex::new(r"&").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from("|"), rule: Some(Regex::new(r"\|").unwrap()), category: TokenCategory::Delimiter },
        TokenStruct { word: String::from("^"), rule: Some(Regex::new(r"\^").unwrap()), category: TokenCategory::Delimiter },

        TokenStruct { word: String::from("\\n"), rule: Some(Regex::new(r"\n").unwrap()), category: TokenCategory::Indentation },
        TokenStruct { word: String::from("indent"), rule: Some(Regex::new(r"^[ \t]+").unwrap()), category: TokenCategory::Indentation },
        TokenStruct { word: String::from("dedent"), rule: Some(Regex::new(r"$^").unwrap()), category: TokenCategory::Indentation },
    ]
}

pub fn regex_match(input:String)->Vec<TokenStruct>{

    let unknown_token= TokenStruct{
        word:String::from("unknown"),
        rule:None,
        category: TokenCategory::Unknown
    };

    let regexes= regex_list();
    let mut token_references= vec![unknown_token;input.len()];
    let mut counter=1;
    
    for rule in regexes {
        match rule.rule {
            Some(ref regex) => {
                for m in regex.find_iter(&input) {
                    println!("Hello the result is, {}:{}", m.start(), m.end());
                    token_references[m.start()] = rule.clone();
                    println!("count {}", counter);
                    counter += 1;
                }
            }
            None => {
                println!("No regex rule for {}", rule.word);
            }
        }
    }

    return token_references;

}