use engine::phisics::PhysicsWorld;

fn main() {
    let mut physics_world = PhysicsWorld::new();

    // Run the simulation
    for step in 0..=200 {
        if step % 20 == 0 {
            println!("Step {}", step);
            physics_world.print_arm_state();

            // Print tricep's farthest corners
            if let Some(((upper_x, upper_y), (lower_x, lower_y))) =
                physics_world.tricep_farthest_corners()
            {
                println!(
                    "Tricep farthest corners: upper=({:.3}, {:.3}), lower=({:.3}, {:.3})",
                    upper_x, upper_y, lower_x, lower_y
                );
            } else {
                println!("Tricep farthest corners: Could not calculate");
            }

            // Print forearm's farthest corners
            if let Some(((upper_x, upper_y), (lower_x, lower_y))) =
                physics_world.forearm_farthest_corners()
            {
                println!(
                    "Forearm farthest corners: upper=({:.3}, {:.3}), lower=({:.3}, {:.3})",
                    upper_x, upper_y, lower_x, lower_y
                );
            } else {
                println!("Forearm farthest corners: Could not calculate");
            }

            physics_world.print_ball_state();
        }

        // Apply a small force to the tricep to create movement
        if step < 50 {
            physics_world.apply_tricep_force(0.4); // 40% force away from wall
        }

        physics_world.step();
    }
}