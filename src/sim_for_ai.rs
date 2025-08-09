use burn::prelude::{Backend, Tensor};
use crate::base_ai::AI;
use crate::phisics::{normalize_x, normalize_y, PhysicsWorld};

type Corners = Option<((f32, f32), (f32, f32))>;

fn add_to_input(tensor_input: &mut Vec<f32>, corners: Corners) {
    let corners = corners.expect("corners not found");
    for coord in [corners.0.0, corners.0.1, corners.1.0, corners.1.1] {
        tensor_input.push(coord);
    }
}

fn add_to_input_normalized(
    tensor_input: &mut Vec<f32>,
    corners: Corners,
) {
    let corners = corners.expect("normalized corners not found");
    for corner in [corners.0, corners.1] {
        tensor_input.push(normalize_x(corner.0));
        tensor_input.push(normalize_y(corner.1));
    }
}

fn saved_to_both(
    tensor_input: &mut Vec<f32>,
    saved_corners: &mut Vec<f32>,
    corners: Corners,
) {
    add_to_input_normalized(tensor_input, corners);
    add_to_input_normalized(saved_corners, corners);
}

fn capture_world_state(world: &PhysicsWorld) -> [Corners; 7] {
    [world.tricep_farthest_corners(),
        world.forearm_farthest_corners(), world.palm_farthest_corners(),
        world.lower_index_finger_farthest_corners(), world.upper_index_finger_farthest_corners(),
        world.lower_thumb_farthest_corners(), world.upper_thumb_farthest_corners()]
}

fn on_captured_state<FN>(world: &PhysicsWorld, mut action: FN)
where FN: FnMut(Corners)
{
    for corners in capture_world_state(world) {
        action(corners);
    }
}

fn save_world_state(world:&PhysicsWorld, save_location:&mut Vec<f32>) {
    on_captured_state(world, |corners| add_to_input(save_location, corners));
}

pub fn test_ai<A, B: Backend>(network: &A, device: &B::Device) -> f32
where
    A: AI<B>,
{
    let mut world = PhysicsWorld::new();

    let mut init_state: Vec<f32> = Vec::new();

    save_world_state(&world, &mut init_state);

    let mut previous_corners = Vec::new();

    on_captured_state(&world, |corners| add_to_input_normalized(&mut previous_corners, corners));

    let mut tensor_input: Vec<f32> = Vec::new();
    let mut saved_steps_scores: Vec<f32> = Vec::new();

    for _ in 0..500 {
        tensor_input.clear();
        tensor_input.extend(&previous_corners);
        previous_corners.clear();

        on_captured_state(&world, |corners| saved_to_both(&mut tensor_input, &mut previous_corners, corners));

        // previous ball x
        tensor_input.push(0.0);
        // previous ball y
        tensor_input.push(0.0);
        // previous distance to basket x
        tensor_input.push(0.0);
        // previous distance to basket y
        tensor_input.push(0.0);

        // ball x
        tensor_input.push(0.0);
        // ball y
        tensor_input.push(0.0);
        // distance to basket x
        tensor_input.push(0.0);
        // distance to basket y
        tensor_input.push(0.0);
        let tensor = Tensor::<B, 1>::from_floats(tensor_input.as_slice(), device);
        let data = network.apply(tensor).to_data();
        let forces: &[f32] = data.as_slice().expect("ai requested forces not available");

        world.apply_tricep_force(forces[0]);
        world.apply_forearm_force(forces[1]);
        world.apply_palm_force(forces[2]);
        world.apply_lower_index_finger_force(forces[3]);
        world.apply_upper_index_finger_force(forces[4]);
        world.apply_lower_thumb_force(forces[5]);
        world.apply_upper_thumb_force(forces[6]);
        world.step();
        saved_steps_scores.push(scorer(&init_state, &world));
    }
    let last_score = *saved_steps_scores.last().expect("saved steps scores empty");

    saved_steps_scores.sort_by(|a, b| a.partial_cmp(b).expect("saved scores not comparable"));

    (saved_steps_scores[saved_steps_scores.len() / 2] * 10.0
        + last_score * 5.
        + saved_steps_scores[0]
        + saved_steps_scores[saved_steps_scores.len() - 1])
        / 17.
}

fn scorer(init_state: &Vec<f32>, world: &PhysicsWorld) -> f32 {
    let mut end_state: Vec<f32> = Vec::new();
    save_world_state(world, &mut end_state);

    let mape = init_state
        .iter()
        .zip(end_state.iter())
        .map(|(a, b)| ((a - b) / a).abs())
        .sum::<f32>()
        / init_state.len() as f32;

    1. / (mape + 1.)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::BigAI;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    use std::time::SystemTime;

    #[test]
    fn test_ai_simulation() {
        type BE = NdArray<f32>;
        let device = NdArrayDevice::Cpu;
        let before = SystemTime::now();
        let treat = test_ai(&BigAI::<BE>::new(&device), &device);
        let time_taken = before.elapsed().unwrap().as_millis();
        println!("Time taken: {} ms", time_taken);

        println!("Treat: {treat}");
    }
}