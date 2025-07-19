use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::tensor::Distribution;
use engine::ai::AI;
use engine::test_ai;
use rand::Rng;
use std::time::SystemTime;

fn main() {
    type BE = NdArray<f32>;
    let device = NdArrayDevice::Cpu;
    let mut all_ais: Vec<_> = (0..200).map(|_| AI::<BE>::new(&device)).collect();
    for i in 0..1000 {
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

        all_ais = make_new_generation(ai_w_scores, &device, i);
    }
}

fn make_new_generation<B: Backend>(
    ais_w_score: Vec<(f32, AI<B>)>,
    device: &B::Device,
    past_generation_count: usize,
) -> Vec<AI<B>> {
    // dont use all parents at one time
    let quarter_generation = (0.25 * ais_w_score.len() as f32) as usize;
    let best_ones: Vec<_> = ais_w_score
        .iter()
        .take(quarter_generation)
        .map(|(_, ai)| ai.clone())
        .collect();
    let mut new_generation = Vec::new();
    new_generation.extend((0..3).map(|_| AI::<B>::new(device)));
    let mut rng = rand::thread_rng();

    let distribution = Distribution::Normal(0.0, 0.05 / (past_generation_count + 1) as f64);

    for _ in 0..(ais_w_score.len() - best_ones.len() - new_generation.len()) {
        let mother = rng.random_range(0..quarter_generation);

        let father = loop {
            let potential_father = rng.random_range(0..quarter_generation);
            if potential_father != mother {
                break potential_father;
            }
        };

        let offspring = match rng.random_range(0..10) {
            0 | 1 | 2 | 3 | 4 | 5 | 6 => {
                best_ones[mother].offspring_iw(&best_ones[father], &distribution)
            }
            7 | 8 => best_ones[mother].offspring(&best_ones[father], &distribution),
            9 => best_ones[mother].jiggle(&distribution),

            _ => unreachable!(),
        };
        new_generation.push(offspring);
    }

    new_generation.extend(best_ones);

    new_generation
}
