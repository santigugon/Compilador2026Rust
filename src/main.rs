mod read_file;
mod regex;
mod enums;
mod models;
mod transition_matching;

use regex::utils;
use transition_matching::automata;
use enums::transition_tables::States;

fn main() {
    println!("Hello, world!");
    // read_file::read_input();
    // utils::regex_match();
    let result = automata::state_match('h',States::Q0);
    // println!("The result is ,{}", result); 
}
