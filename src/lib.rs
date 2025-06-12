use rapier2d::prelude::*;

pub fn create_physics_world() {
    let mut rigid_body_set = RigidBodySet::new();
    let mut collider_set = ColliderSet::new();
    let mut impulse_joint_set = ImpulseJointSet::new();

    // Create the ground
    let ground_rigid_body = RigidBodyBuilder::fixed()
        .translation(vector![0.0, -2.0])
        .build();
    let ground_handle = rigid_body_set.insert(ground_rigid_body);

    let ground_collider = ColliderBuilder::cuboid(100.0, 0.1)
        .restitution(0.7)
        .friction(0.3)
        .build();
    collider_set.insert_with_parent(ground_collider, ground_handle, &mut rigid_body_set);

    // Calculate starting positions (just above the ground)
    let ground_top = -2.0 + 0.1; // Ground y position + ground height
    let box1_height = 1.0; // Height of box1
    let box2_height = 0.5; // Height of box2
    let box1_start_y = ground_top + (box1_height / 2.0);
    
    // Create the first (larger) box
    let box1_body = RigidBodyBuilder::dynamic()
        .translation(vector![0.0, box1_start_y])
        .can_sleep(false)
        .ccd_enabled(true)
        .build();
    let box1_handle = rigid_body_set.insert(box1_body);

    let box1_collider = ColliderBuilder::cuboid(0.5, 0.5)
        .restitution(0.7)
        .friction(0.3)
        .active_events(ActiveEvents::COLLISION_EVENTS)
        .build();
    collider_set.insert_with_parent(box1_collider, box1_handle, &mut rigid_body_set);

    // Create the second (smaller) box
    let box2_body = RigidBodyBuilder::dynamic()
        .translation(vector![1.2, box1_start_y])
        .can_sleep(false)
        .ccd_enabled(true)
        .build();
    let box2_handle = rigid_body_set.insert(box2_body);

    let box2_collider = ColliderBuilder::cuboid(0.25, 0.25)
        .restitution(0.7)
        .friction(0.3)
        .active_events(ActiveEvents::COLLISION_EVENTS)
        .build();
    collider_set.insert_with_parent(box2_collider, box2_handle, &mut rigid_body_set);

    // Create a revolute joint between the boxes
    let joint = RevoluteJointBuilder::new()
        .local_anchor1(point![0.6, 0.0])
        .local_anchor2(point![-0.3, 0.0])
        .build();
    impulse_joint_set.insert(box1_handle, box2_handle, joint, true);

    let gravity = vector![0.0, -9.81];
    let mut integration_parameters = IntegrationParameters::default();
    integration_parameters.dt = 1.0 / 240.0;
    integration_parameters.max_ccd_substeps = 4;

    let mut physics_pipeline = PhysicsPipeline::new();
    let mut island_manager = IslandManager::new();
    let mut broad_phase = BroadPhase::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut multibody_joint_set = MultibodyJointSet::new();
    let mut ccd_solver = CCDSolver::new();

    let physics_hooks = ();
    let event_handler = ();

    // Run the simulation
    for step in 0..100 {
        // Apply constant force to box1
        if let Some(rb1) = rigid_body_set.get_mut(box1_handle) {
            // Apply force in positive x direction
            rb1.add_force(vector![5.0, 0.0], true);
        }

        physics_pipeline.step(
            &gravity,
            &integration_parameters,
            &mut island_manager,
            &mut broad_phase,
            &mut narrow_phase,
            &mut rigid_body_set,
            &mut collider_set,
            &mut impulse_joint_set,
            &mut multibody_joint_set,
            &mut ccd_solver,
            None,
            &physics_hooks,
            &event_handler,
        );

        // Print positions and velocities of both boxes
        println!("Step {}", step);
        
        if let Some(rb1) = rigid_body_set.get(box1_handle) {
            let pos1 = rb1.translation();
            let vel1 = rb1.linvel();
            println!("Box1 center: x={:.3}, y={:.3}, vel_x={:.3}", pos1.x, pos1.y, vel1.x);
        }
        
        if let Some(rb2) = rigid_body_set.get(box2_handle) {
            let pos2 = rb2.translation();
            let vel2 = rb2.linvel();
            println!("Box2 center: x={:.3}, y={:.3}, vel_x={:.3}\n", pos2.x, pos2.y, vel2.x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_simulation() {
        create_physics_world();
    }
}