use std::fs;

pub fn read_input(){
    println!("HOLA MODULO");
    let file_path="input.txt";

    println!("In file {file_path}");

    let contents = fs::read_to_string(file_path).expect("Should have been able to read file");

    println!("With text:\n{contents}");
}