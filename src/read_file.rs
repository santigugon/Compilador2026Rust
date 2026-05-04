use std::fs;

pub fn read_input()->String{
    let file_path="input.txt";
    let mut contents = fs::read_to_string(file_path).expect("Should have been able to read file");
    contents.push(' ');

    return contents;
}