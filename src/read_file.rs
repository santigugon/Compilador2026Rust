use std::fs;
use std::path::Path;

pub fn read_input(path: &Path) -> String {
    let mut contents = fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Should have been able to read file {}", path.display()));
    contents.push(' ');

    contents
}
