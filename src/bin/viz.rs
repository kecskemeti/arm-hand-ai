use burn::backend::candle::CandleDevice;
use burn::backend::Candle;
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use engine::base_ai::ListableAI;
use engine::base_ai::AI;
use engine::sim_for_ai::visual_ai;
use engine::{ai, small_ai};

type BE = Candle<f32, i64>;

fn small_ai_maker<BE: Backend>(d: &BE::Device) -> impl ListableAI<BE> {
    small_ai::SmallAI::<BE>::new(d)
}

fn big_ai_maker<BE: Backend>(d: &BE::Device) -> impl ListableAI<BE> {
    ai::BigAI::<BE>::new(d)
}

fn run_viz<B: Backend, A: AI<B>>(
    ai_maker: &impl Fn(&B::Device) -> A,
    mpk_name: &str,
    device: &B::Device,
) {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let sample_ai = ai_maker(&device);

    let actual_ai = sample_ai.load_a_file(mpk_name, &recorder);
    visual_ai(&actual_ai, &device);
}

fn main() {
    let device = CandleDevice::Cpu;

    let args = std::env::args().collect::<Vec<_>>();

    let mpk_name = args[1].clone();
    let big = big_ai_maker::<BE>(&device);
    let small = small_ai_maker::<BE>(&device);
    if mpk_name.contains(big.network_name()) {
        run_viz(&big_ai_maker::<BE>, &mpk_name, &device);
    } else if mpk_name.contains(small.network_name()) {
        run_viz(&small_ai_maker::<BE>, &mpk_name, &device);
    } else {
        panic!("Invalid network name");
    }
}
