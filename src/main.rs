mod read_file;
mod regex;
mod enums;
mod models;
mod transition_matching;
mod smash;

use regex::utils;
use transition_matching::automata;
use enums::transition_tables::States;
use smash::merge_lists;

fn main() {
    println!("Hello, world!");
    // let input =String::from("/while if ifa whilea +-= 44 4.2898 asd4 ");
    let  input: String = read_file::read_input();
    let automata_tokens= automata::match_transitions(&input);
    let used_positions =automata::return_used_positions(&automata_tokens);
    let regex_list =utils::regex_match(input);

    let smashed_list = merge_lists::automata_regex_match(&automata_tokens,&used_positions, &regex_list);

    for token in smashed_list{
        println!("FINAL word#{}#{}#", token.word, token.category);
    }

    //  for i in 0..(used_position.len()-1){
    //     println!("Position {}:{}", i, used_position[i]);
    //  }

    // println!("The result is ,{}", result); 
}
