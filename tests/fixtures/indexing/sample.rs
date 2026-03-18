pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub struct Greeter {
    name: String,
}

impl Greeter {
    pub fn greet(&self) -> String {
        format!("Hello {}", self.name)
    }
}

pub fn long_calc(input: i32) -> i32 {
    let mut acc = input;
    acc += 1;
    acc += 2;
    acc += 3;
    acc += 4;
    acc += 5;
    acc += 6;
    acc += 7;
    acc += 8;
    acc += 9;
    acc += 10;
    acc += 11;
    acc += 12;
    acc += 13;
    acc += 14;
    acc += 15;
    acc += 16;
    acc += 17;
    acc += 18;
    acc += 19;
    acc += 20;
    acc
}
