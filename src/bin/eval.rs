use burn::backend::candle::CandleDevice;
use burn::backend::Candle;
use burn::prelude::Backend;
use burn::tensor::Distribution;

use engine::ai;
use engine::base_ai::AI;
use engine::sim_for_ai::{test_ai, visual_ai};
use rayon::prelude::*;
use std::time::SystemTime;

static BEST_PROPORTION: f32 = 0.25;
static ISLAND_POPULATION: usize = 100;
static ALWAYS_RAND_COUNT: usize = 3;

static SMALLEST_SD: f64 = 0.01;
type BE = Candle<f32, i64>;

fn ai_maker<BE: Backend>(d: &BE::Device) -> impl AI<BE> {
    ai::BigAI::<BE>::new(d)
}

fn main() {
    let device = CandleDevice::Cpu;

    let mut islands: [Vec<_>; 5] = (0..5)
        .map(|_| {
            (0..ISLAND_POPULATION)
                .map(|_| ai_maker::<BE>(&device))
                .collect()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let mut best_score = 0.0;

    for i in 0..10000 {
        for (j, island) in islands.iter_mut().enumerate() {
            let before = SystemTime::now();
            let inner_ais = island.clone();
            let mut ai_w_scores = inner_ais
                .into_par_iter()
                .map(|ai| (test_ai(&ai, &device), ai))
                .collect::<Vec<_>>();
            ai_w_scores.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .expect("ai score should be comparable")
            });

            let time_taken = before.elapsed().expect("elapsed calc failed").as_millis();
            println!("{i},{j} Time taken: {} ms", time_taken);

            let high_score = ai_w_scores
                .iter()
                .next()
                .map(|(score, _)| *score)
                .expect("high score not found");
            if high_score > best_score {
                best_score = high_score;
                println!("{i},{j} New best score: {}", high_score);
                visual_ai(&ai_w_scores[0].1, &device);
            }
            println!("{i},{j} Best score: {}", high_score);
            println!("{i},{j} Best mape: {}", (1.0 / high_score) - 1.);

            *island = make_new_generation(ai_w_scores, &device, BEST_PROPORTION, ai_maker);
        }

        if i % 30 == 0 {
            island_crossing(&mut islands);
        }
    }
}

pub fn island_crossing<B: Backend, A: AI<B>>(islands: &mut [Vec<A>; 5]) {
    let fittest_start = ISLAND_POPULATION - (ISLAND_POPULATION as f32 * BEST_PROPORTION) as usize;
    // Clone the best individuals instead of holding references
    let best: Vec<Vec<A>> = islands
        .iter()
        .map(|island| island[fittest_start..].to_vec())
        .collect();

    let island_count = islands.len();
    let fittest_count = best[0].len();

    // 10 crossings
    for _ in 0..10 {
        let (mothers_island, fathers_island) = make_distinct(island_count);

        let offspring = {
            let mother = &best[mothers_island][rand::random_range(0..fittest_count)];
            let father = &best[fathers_island][rand::random_range(0..fittest_count)];
            make_offspring(mother, father, &Distribution::Normal(0.0, SMALLEST_SD))
        };
        islands[mothers_island][rand::random_range(0..ALWAYS_RAND_COUNT)] = offspring;
    }
}

fn make_new_generation<B: Backend, A: AI<B>>(
    ais_w_score: Vec<(f32, A)>,
    device: &B::Device,
    best_proportion: f32,
    ai_maker: impl Fn(&B::Device) -> A,
) -> Vec<A> {
    let best_score = ais_w_score[0].0;
    let std_deviation = ais_w_score[0].1.max_amp() as f64
        * if best_score < 0.5 {
            0.15
        } else if best_score < 0.75 {
            0.075
        } else if best_score < 0.9 {
            0.05
        } else if best_score < 0.95 {
            0.02
        } else {
            SMALLEST_SD
        };

    let distribution = Distribution::Normal(0.0, std_deviation);

    // don't keep parents once they are combined.
    let number_of_fittest = (best_proportion * ais_w_score.len() as f32) as usize;
    let best_ones: Vec<_> = ais_w_score
        .iter()
        .take(number_of_fittest)
        .map(|(_, ai)| ai.clone())
        .collect();
    let mut new_generation = Vec::new();
    new_generation.extend((0..ALWAYS_RAND_COUNT).map(|_| ai_maker(device)));

    for _ in 0..(ais_w_score.len() - best_ones.len() - new_generation.len()) {
        let (mother, father) = make_distinct(number_of_fittest);
        let offspring = make_offspring(&best_ones[mother], &best_ones[father], &distribution);
        new_generation.push(offspring);
    }

    new_generation.extend(best_ones);

    new_generation
}

pub fn make_distinct(max: usize) -> (usize, usize) {
    let specimen_one = rand::random_range(0..max);

    let specimen_two = loop {
        let potential_specimen = rand::random_range(0..max);
        if potential_specimen != specimen_one {
            break potential_specimen;
        }
    };

    (specimen_one, specimen_two)
}

pub fn make_offspring<B: Backend, A: AI<B>>(
    mother: &A,
    father: &A,
    distribution: &Distribution,
) -> A {
    match rand::random_range(0..15) {
        0 | 1 | 2 | 3 | 4 => mother.offspring_iw(father, distribution),
        5 | 6 | 7 | 8 => mother.offspring_aw(father, distribution),
        9 => mother.offspring(father, distribution),
        10 => mother.offspring_layers(father, distribution),
        11 | 12 => mother.jiggle(distribution),
        13 | 14 => father.jiggle(distribution),
        _ => unreachable!(),
    }
}
