mod read_file;
mod regex;
mod enums;
mod models;
mod transition_matching;
mod smash;
mod semantic_analysis;
mod lexical_diagnostics;
mod parser;
mod ast;

use std::fs;
use std::path::{Path, PathBuf};

use regex::utils;
use transition_matching::automata;
use smash::merge_lists;
use models::token_model::TokenStruct;
use parser::Parser;

const LEXEME_COLUMN_WIDTH: usize = 32;

struct TestCase {
    path: PathBuf,
    expected_valid: bool,
}


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

fn status_label(is_valid: bool) -> &'static str {
    if is_valid { "VALIDO" } else { "INVALIDO" }
}

fn read_case_input(path: &Path) -> String {
    let mut contents = fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Should have been able to read {}", path.display()));
    contents.push(' ');
    contents
}

fn collect_test_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();

    for (folder, expected_valid) in [("valid", true), ("invalid", false)] {
        let dir = Path::new("test_inputs").join(folder);
        if !dir.exists() {
            continue;
        }

        let mut folder_cases: Vec<TestCase> = fs::read_dir(&dir)
            .unwrap_or_else(|_| panic!("Should have been able to read {}", dir.display()))
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                if path.is_file() {
                    Some(TestCase { path, expected_valid })
                } else {
                    None
                }
            })
            .collect();

        folder_cases.sort_by(|a, b| a.path.cmp(&b.path));
        cases.extend(folder_cases);
    }

    cases
}

fn run_case(case_name: &str, expected_valid: bool, input: String) -> bool {
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

    println!();
    println!("{}", "=".repeat(80));
    println!("Caso: {case_name}");
    println!("Esperado: {}", status_label(expected_valid));

    let semantic_ok = match semantic_analysis::validate_operator_operands(&smashed_list) {
        Ok(()) => {
            println!("Semantic analysis: OK (operators have valid neighbors).");
            true
        }
        Err(messages) => {
            eprintln!("Semantic analysis: failed.");
            for msg in &messages {
                eprintln!("  {msg}");
            }
            false
        }
    };

    if !lexical_issues.is_empty() {
        eprintln!("Lexical analysis: failed.");
        for issue in &lexical_issues {
            eprintln!("  '{}' — {}", issue.lexeme, issue.message);
        }
    } else {
        print_tokens(&smashed_list);
    }

    let mut p = Parser::new(smashed_list.clone());
    let parse_ok = match p.parse_program() {
        Ok(ast) => {
            println!("VALIDO");
            println!("{}", ast.pretty(0));
            true
        }
        Err(e) => {
            println!("INVALIDO");
            eprintln!("{}", e);
            false
        }
    };

    let actual_valid = semantic_ok && lexical_issues.is_empty() && parse_ok;
    let matches_expectation = actual_valid == expected_valid;
    println!("Resultado final: {}", status_label(actual_valid));
    println!(
        "Coincide con lo esperado: {}",
        if matches_expectation { "SI" } else { "NO" }
    );

    matches_expectation
}

fn main() {
    let test_cases = collect_test_cases();

    if test_cases.is_empty() {
        let input: String = read_file::read_input();
        let _ = run_case("input.txt", true, input);
        return;
    }

    let mut matching_cases = 0;
    let total_cases = test_cases.len();

    for case in test_cases {
        let case_name = case.path.display().to_string();
        let input = read_case_input(&case.path);
        if run_case(&case_name, case.expected_valid, input) {
            matching_cases += 1;
        }
    }

    println!();
    println!("{}", "=".repeat(80));
    println!(
        "Resumen: {matching_cases}/{total_cases} casos coincidieron con el resultado esperado."
    );
}
