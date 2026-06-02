mod ast;
mod enums;
mod lexical_diagnostics;
mod models;
mod parser;
mod read_file;
mod regex;
mod semantic_analysis;
mod smash;
mod transition_matching;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use models::token_model::TokenStruct;
use parser::Parser;
use regex::utils;
use smash::merge_lists;
use transition_matching::automata;

const LEXEME_COLUMN_WIDTH: usize = 32;

struct TestCase {
    path: PathBuf,
    expected_valid: bool,
}

struct RunSummary {
    actual_valid: bool,
    matches_expectation: Option<bool>,
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

fn format_char_position(position: Option<usize>) -> String {
    match position {
        Some(index) => format!(" [char {}]", index),
        None => String::new(),
    }
}

fn read_case_input(path: &Path) -> String {
    read_file::read_input(path)
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
                    Some(TestCase {
                        path,
                        expected_valid,
                    })
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

fn run_case(case_name: &str, expected_valid: Option<bool>, input: String) -> RunSummary {
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
    if let Some(expected_valid) = expected_valid {
        println!("Esperado: {}", status_label(expected_valid));
    }

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
            eprintln!(
                "  '{}'{} — {}",
                issue.lexeme,
                format_char_position(issue.char_index),
                issue.message
            );
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
    println!("Resultado final: {}", status_label(actual_valid));
    let matches_expectation = expected_valid.map(|expected| actual_valid == expected);
    if let Some(matches_expectation) = matches_expectation {
        println!(
            "Coincide con lo esperado: {}",
            if matches_expectation { "SI" } else { "NO" }
        );
    }

    RunSummary {
        actual_valid,
        matches_expectation,
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    if let Some(first_arg) = args.first() {
        if first_arg == "--run-tests" {
            let test_cases = collect_test_cases();
            if test_cases.is_empty() {
                eprintln!("No se encontraron casos en test_inputs/.");
                return;
            }

            let mut matching_cases = 0;
            let total_cases = test_cases.len();

            for case in test_cases {
                let case_name = case.path.display().to_string();
                let input = read_case_input(&case.path);
                let summary = run_case(&case_name, Some(case.expected_valid), input);
                if summary.matches_expectation == Some(true) {
                    matching_cases += 1;
                }
            }

            println!();
            println!("{}", "=".repeat(80));
            println!(
                "Resumen: {matching_cases}/{total_cases} casos coincidieron con el resultado esperado."
            );
            return;
        }

        let path = Path::new(first_arg);
        let input = read_case_input(path);
        let summary = run_case(&path.display().to_string(), None, input);
        std::process::exit(if summary.actual_valid { 0 } else { 1 });
    }

    let default_input = Path::new("input.txt");
    if default_input.exists() {
        let input = read_case_input(default_input);
        let summary = run_case("input.txt", None, input);
        std::process::exit(if summary.actual_valid { 0 } else { 1 });
    }

    let test_cases = collect_test_cases();
    if test_cases.is_empty() {
        eprintln!("Uso: mini_triton_parser <archivo.mt> o mini_triton_parser --run-tests");
        return;
    }

    let mut matching_cases = 0;
    let total_cases = test_cases.len();

    for case in test_cases {
        let case_name = case.path.display().to_string();
        let input = read_case_input(&case.path);
        let summary = run_case(&case_name, Some(case.expected_valid), input);
        if summary.matches_expectation == Some(true) {
            matching_cases += 1;
        }
    }

    println!();
    println!("{}", "=".repeat(80));
    println!(
        "Resumen: {matching_cases}/{total_cases} casos coincidieron con el resultado esperado."
    );
}
