use burn::backend::candle::CandleDevice;
use burn::backend::Candle;
use burn::prelude::Backend;
use burn::tensor::Distribution;
use engine::ai::AI;
use engine::test_ai;
use rand::Rng;
use rayon::prelude::*;
use std::time::SystemTime;

static BEST_PROPORTION: f32 = 0.25;
static ISLAND_POPULATION: usize = 100;

static SMALLEST_SD: f64 = 0.01;

fn main() {
    // experiment with island concept when mixing
    type BE = Candle<f32, i64>;
    let device = CandleDevice::Cpu;

    let mut islands: [Vec<_>; 5] = [
        (0..ISLAND_POPULATION)
            .map(|_| AI::<BE>::new(&device))
            .collect(),
        (0..ISLAND_POPULATION)
            .map(|_| AI::<BE>::new(&device))
            .collect(),
        (0..ISLAND_POPULATION)
            .map(|_| AI::<BE>::new(&device))
            .collect(),
        (0..ISLAND_POPULATION)
            .map(|_| AI::<BE>::new(&device))
            .collect(),
        (0..ISLAND_POPULATION)
            .map(|_| AI::<BE>::new(&device))
            .collect(),
    ];

    for i in 0..1000 {
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
            println!("{i},{j} Best score: {}", high_score);
            println!("{i},{j} Best mape: {}", (1.0 / high_score) - 1.);

            *island = make_new_generation(ai_w_scores, &device, BEST_PROPORTION);
        }

        if i % 30 == 0 {
            island_crossing(&mut islands);
        }
    }
}

pub fn island_crossing<B: Backend>(islands: &mut [Vec<AI<B>>; 5]) {
    let mut rng = rand::thread_rng();

    let fittest_start = crate::ISLAND_POPULATION
        - (crate::ISLAND_POPULATION as f32 * crate::BEST_PROPORTION) as usize;
    // Clone the best individuals instead of holding references
    let best: Vec<Vec<AI<B>>> = islands
        .iter()
        .map(|island| island[fittest_start..].to_vec())
        .collect();

    let island_count = islands.len();
    let fittest_count = best[0].len();

    // 10 crossings
    for _ in 0..10 {
        let (mothers_island, fathers_island) = make_distinct(island_count);

        let offspring = {
            let mother = &best[mothers_island][rng.random_range(0..fittest_count)];
            let father = &best[fathers_island][rng.random_range(0..fittest_count)];
            make_offspring(mother, father, &Distribution::Normal(0.0, SMALLEST_SD))
        };
        islands[mothers_island][0] = offspring;
    }
}

fn make_new_generation<B: Backend>(
    ais_w_score: Vec<(f32, AI<B>)>,
    device: &B::Device,
    best_proportion: f32,
) -> Vec<AI<B>> {
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
    new_generation.extend((0..3).map(|_| AI::<B>::new(device)));

    for _ in 0..(ais_w_score.len() - best_ones.len() - new_generation.len()) {
        let (mother, father) = make_distinct(number_of_fittest);
        let offspring = make_offspring(&best_ones[mother], &best_ones[father], &distribution);
        new_generation.push(offspring);
    }

    new_generation.extend(best_ones);

    new_generation
}

pub fn make_distinct(max: usize) -> (usize, usize) {
    let mut rng = rand::thread_rng();

    let specimen_one = rng.random_range(0..max);

    let specimen_two = loop {
        let potential_specimen = rng.random_range(0..max);
        if potential_specimen != specimen_one {
            break potential_specimen;
        }
    };

    (specimen_one, specimen_two)
}

pub fn make_offspring<B: Backend>(
    mother: &AI<B>,
    father: &AI<B>,
    distribution: &Distribution,
) -> AI<B> {
    let mut rng = rand::thread_rng();
    match rng.random_range(0..15) {
        0 | 1 | 2 | 3 | 4 => mother.offspring_iw(father, distribution),
        5 | 6 | 7 | 8 => mother.offspring_aw(father, distribution),
        9 => mother.offspring(father, distribution),
        10 => mother.offspring_layers(father, distribution),
        11 | 12 => mother.jiggle(distribution),
        13 | 14 => father.jiggle(distribution),
        _ => unreachable!(),
    }
}
