use burn::backend::candle::CandleDevice;
use burn::backend::Candle;
use burn::prelude::Backend;
use burn::tensor::Distribution;
use engine::ai::AI;
use engine::test_ai;
use rand::Rng;
use rayon::prelude::*;
use std::time::SystemTime;

fn main() {
    // experiment with island concept when mixing
    type BE = Candle<f32, i64>;
    let device = CandleDevice::Cpu;

    let mut islands: [Vec<_>; 5] = [
        (0..100).map(|_| AI::<BE>::new(&device)).collect(),
        (0..100).map(|_| AI::<BE>::new(&device)).collect(),
        (0..100).map(|_| AI::<BE>::new(&device)).collect(),
        (0..100).map(|_| AI::<BE>::new(&device)).collect(),
        (0..100).map(|_| AI::<BE>::new(&device)).collect(),
    ];

    for i in 0..15 {
        for (j, island) in islands.iter_mut().enumerate() {
            let before = SystemTime::now();
            let inner_ais = island.clone();
            let mut ai_w_scores = inner_ais
                .into_par_iter()
                .map(|ai| (test_ai(&ai, &device), ai))
                .collect::<Vec<_>>();
            ai_w_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            let time_taken = before.elapsed().unwrap().as_millis();
            println!("{i},{j} Time taken: {} ms", time_taken);

            let high_score = ai_w_scores.iter().next().map(|(score, _)| *score).unwrap();
            println!("{i},{j} Best score: {}", high_score);
            println!("{i},{j} Best mape: {}", (1.0 / high_score) - 1.);

            *island = make_new_generation(ai_w_scores, &device, i);
        }
    }
}

fn make_new_generation<B: Backend>(
    ais_w_score: Vec<(f32, AI<B>)>,
    device: &B::Device,
    past_generation_count: usize,
) -> Vec<AI<B>> {
    let best_score = ais_w_score[0].0;
    let std_deviation = ais_w_score[0].1.max_amp()
        * if best_score < 0.5 {
            0.15
        } else if best_score < 0.75 {
            0.075
        } else if best_score < 0.9 {
            0.05
        } else if best_score < 0.95 {
            0.02
        } else {
            0.01
        };

    let distribution = Distribution::Normal(0.0, std_deviation as f64);

    // don't keep parents once they are combined.
    let quarter_generation = (0.25 * ais_w_score.len() as f32) as usize;
    let best_ones: Vec<_> = ais_w_score
        .iter()
        .take(quarter_generation)
        .map(|(_, ai)| ai.clone())
        .collect();
    let mut new_generation = Vec::new();
    new_generation.extend((0..3).map(|_| AI::<B>::new(device)));
    let mut rng = rand::thread_rng();

    for _ in 0..(ais_w_score.len() - best_ones.len() - new_generation.len()) {
        let mother = rng.random_range(0..quarter_generation);

        let father = loop {
            let potential_father = rng.random_range(0..quarter_generation);
            if potential_father != mother {
                break potential_father;
            }
        };

        let offspring = match rng.random_range(0..15) {
            0 | 1 | 2 | 3 | 4 => best_ones[mother].offspring_iw(&best_ones[father], &distribution),
            5 | 6 | 7 | 8 => best_ones[mother].offspring_aw(&best_ones[father], &distribution),
            9 => best_ones[mother].offspring(&best_ones[father], &distribution),
            10 => best_ones[mother].offspring_layers(&best_ones[father], &distribution),
            11 | 12 => best_ones[mother].jiggle(&distribution),
            13 | 14 => best_ones[father].jiggle(&distribution),
            _ => unreachable!(),
        };
        new_generation.push(offspring);
    }

    new_generation.extend(best_ones);

    new_generation
}
