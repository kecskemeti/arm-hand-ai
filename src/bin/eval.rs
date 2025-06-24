use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use engine::ai::AI;
use engine::test_ai;
use std::time::SystemTime;

fn main() {
    type BE = NdArray<f32>;
    let device = NdArrayDevice::Cpu;
    let before = SystemTime::now();
    test_ai(&AI::<BE>::new(&device), &device);
    let time_taken = before.elapsed().unwrap().as_millis();
    println!("Time taken: {} ms", time_taken);
}
