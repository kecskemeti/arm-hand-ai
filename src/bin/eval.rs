use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::prelude::Backend;
use engine::ai::AI;
use engine::test_ai;
use rand::Rng;
use std::time::SystemTime;

fn main() {
    type BE = NdArray<f32>;
    let device = NdArrayDevice::Cpu;
    let mut all_ais: Vec<_> = (0..200).map(|_| AI::<BE>::new(&device)).collect();
    for i in 0..1500 {
        let before = SystemTime::now();
        let inner_ais = all_ais.clone();
        let mut ai_w_scores = inner_ais
            .into_iter()
            .map(|ai| (test_ai(&ai, &device), ai))
            .collect::<Vec<_>>();
        ai_w_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let time_taken = before.elapsed().unwrap().as_millis();
        println!("{i} Time taken: {} ms", time_taken);

        let high_score = ai_w_scores.iter().next().map(|(score, _)| *score).unwrap();
        println!("{i} Best score: {}", high_score);
        println!("{i} Best reciprocal: {}", 1.0 / high_score);

        all_ais = make_new_generation(ai_w_scores, &device);
    }
}

fn make_new_generation<B: Backend>(
    ais_w_score: Vec<(f32, AI<B>)>,
    device: &B::Device,
) -> Vec<AI<B>> {
    let five_percent = (0.05 * ais_w_score.len() as f32) as usize;
    let best_ones: Vec<_> = ais_w_score
        .iter()
        .take(five_percent)
        .map(|(_, ai)| ai.clone())
        .collect();
    let mut new_generation = Vec::new();
    new_generation.extend((0..five_percent).map(|_| AI::<B>::new(device)));
    let mut rng = rand::thread_rng();
    for _ in 0..(ais_w_score.len() - 2 * five_percent) {
        let mother = rng.random_range(0..five_percent);

        let father = loop {
            let potential_father = rng.random_range(0..five_percent);
            if potential_father != mother {
                break potential_father;
            }
        };

        new_generation.push(best_ones[mother].offspring(&best_ones[father]));
    }

    new_generation.extend(best_ones);

    new_generation
}
