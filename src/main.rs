mod read_file;
mod regex;
mod enums;
mod models;
mod transition_matching;
mod smash;
mod semantic_analysis;
mod lexical_diagnostics;

use regex::utils;
use transition_matching::automata;
use smash::merge_lists;
use models::token_model::TokenStruct;

const LEXEME_COLUMN_WIDTH: usize = 32;


fn print_tokens(tokens: &[TokenStruct]) {
    println!();
    println!("Lexical tokens");
    let rule_len = LEXEME_COLUMN_WIDTH + 4 + 24;
    println!("{}", "─".repeat(rule_len));
    println!(
        "{:<LEX_WIDTH$}  {}",
        "Lexeme",
        "Token",
        LEX_WIDTH = LEXEME_COLUMN_WIDTH
    );
    println!("{}", "─".repeat(rule_len));
    for token in tokens {
        println!(
            "{:<LEX_WIDTH$}  {}",
            token.word,
            token.category,
            LEX_WIDTH = LEXEME_COLUMN_WIDTH
        );
    }
    println!("{}", "─".repeat(rule_len));
}

fn main() {
    // println!("Hello, world!");
    // let input =String::from("/while if ifa whilea +-= 44 4.2898 asd4 ");
    let  input: String = read_file::read_input();
    let (automata_tokens, automata_issues) = automata::match_transitions(&input);
    let used_positions = automata::return_used_positions(&automata_tokens);
    let regex_list = utils::regex_match(input.clone());

    let (smashed_list, merge_issues, raw_merged) =
        merge_lists::automata_regex_match(&automata_tokens, &used_positions, &regex_list, &input);

    let lexical_issues = lexical_diagnostics::consolidate_lexical_issues(
        automata_issues,
        merge_issues,
        &raw_merged,
        &regex_list,
    );

    match semantic_analysis::validate_operator_operands(&smashed_list) {
        Ok(()) => println!("Semantic analysis: OK (operators have valid neighbors)."),
        Err(messages) => {
            eprintln!("Semantic analysis: failed.");
            for msg in &messages {
                eprintln!("  {msg}");
            }
        }
    }

    if !lexical_issues.is_empty() {
        eprintln!("Lexical analysis: failed.");
        for issue in &lexical_issues {
            eprintln!("  '{}' — {}", issue.lexeme, issue.message);
        }
    } else {
        print_tokens(&smashed_list);
    }

    //  for i in 0..(used_position.len()-1){
    //     println!("Position {}:{}", i, used_position[i]);
    //  }

    // println!("The result is ,{}", result); 
}
