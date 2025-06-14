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

    // Create a wall standing on the floor (properly positioned)
    let ground_top = -2.0 + 0.1; // Ground surface at y = -1.9
    let wall_height = 2.0;
    let wall_center_y = ground_top + wall_height; // Wall center at y = 0.1
    
    let wall_rigid_body = RigidBodyBuilder::fixed()
        .translation(vector![0.0, wall_center_y])
        .build();
    let wall_handle = rigid_body_set.insert(wall_rigid_body);

    let wall_collider = ColliderBuilder::cuboid(0.2, wall_height) // Thin wall, 4 units tall
        .restitution(0.7)
        .friction(0.3)
        .build();
    collider_set.insert_with_parent(wall_collider, wall_handle, &mut rigid_body_set);

    // Create the first dynamic box (attached to middle of wall)
    let wall_middle_y = wall_center_y; // Middle of wall is at its center
    let tricep_body = RigidBodyBuilder::dynamic()
        .translation(vector![1.0, wall_middle_y]) // Position it to the right of the wall center
        .can_sleep(false)
        .ccd_enabled(true)
        .build();
    let tricep_handle = rigid_body_set.insert(tricep_body);

    let tricep_collider = ColliderBuilder::cuboid(0.5, 0.5)
        .restitution(0.7)
        .friction(0.3)
        .active_events(ActiveEvents::COLLISION_EVENTS)
        .build();
    collider_set.insert_with_parent(tricep_collider, tricep_handle, &mut rigid_body_set);

    // Create a revolute joint between wall and first box
    let shoulder_joint = RevoluteJointBuilder::new()
        .local_anchor1(point![0.2, 0.0]) // Right edge of wall
        .local_anchor2(point![-0.5, 0.0]) // Left edge of tricep
        .build();
    impulse_joint_set.insert(wall_handle, tricep_handle, shoulder_joint, true);

    // Create the second box (attached to end of first box)
    let forearm_body = RigidBodyBuilder::dynamic()
        .translation(vector![2.0, wall_middle_y]) // Position it to the right of tricep
        .can_sleep(false)
        .ccd_enabled(true)
        .build();
    let forearm_handle = rigid_body_set.insert(forearm_body);

    let forearm_collider = ColliderBuilder::cuboid(0.4, 0.4)
        .restitution(0.7)
        .friction(0.3)
        .active_events(ActiveEvents::COLLISION_EVENTS)
        .build();
    collider_set.insert_with_parent(forearm_collider, forearm_handle, &mut rigid_body_set);

    // Create a revolute joint between first and second box
    let elbow_joint = RevoluteJointBuilder::new()
        .local_anchor1(point![0.5, 0.0]) // Right edge of tricep
        .local_anchor2(point![-0.4, 0.0]) // Left edge of forearm
        .build();
    impulse_joint_set.insert(tricep_handle, forearm_handle, elbow_joint, true);

    // Create the third box (smaller, attached to second box)
    let palm_body = RigidBodyBuilder::dynamic()
        .translation(vector![2.8, wall_middle_y]) // Position it to the right of forearm
        .can_sleep(false)
        .ccd_enabled(true)
        .build();
    let palm_handle = rigid_body_set.insert(palm_body);

    let palm_collider = ColliderBuilder::cuboid(0.25, 0.25) 
        .restitution(0.7)
        .friction(0.3)
        .active_events(ActiveEvents::COLLISION_EVENTS)
        .build();
    collider_set.insert_with_parent(palm_collider, palm_handle, &mut rigid_body_set);

    // Create a revolute joint between second and third box
    let wrist_joint = RevoluteJointBuilder::new()
        .local_anchor1(point![0.4, 0.0]) // Right edge of forearm
        .local_anchor2(point![-0.25, 0.0]) // Left edge of palm
        .build();
    impulse_joint_set.insert(forearm_handle, palm_handle, wrist_joint, true);

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
    for step in 0..=200 {

        if step % 20 == 0 {
            println!("Step {}", step);

            if let Some(rb_tricep) = rigid_body_set.get(tricep_handle) {
                let colliders = rb_tricep.colliders();
                if !colliders.is_empty() {
                    if let Some(collider) = collider_set.get(colliders[0]) {
                        let aabb = collider.compute_aabb();
                        println!("Tricep boundary box: min=({:.3}, {:.3}), max=({:.3}, {:.3})",
                                 aabb.mins.x, aabb.mins.y, aabb.maxs.x, aabb.maxs.y);
                    }
                }
            }

            if let Some(rb_forearm) = rigid_body_set.get(forearm_handle) {
                let colliders = rb_forearm.colliders();
                if !colliders.is_empty() {
                    if let Some(collider) = collider_set.get(colliders[0]) {
                        let aabb = collider.compute_aabb();
                        println!("Forearm boundary box: min=({:.3}, {:.3}), max=({:.3}, {:.3})",
                                 aabb.mins.x, aabb.mins.y, aabb.maxs.x, aabb.maxs.y);
                    }
                }
            }

            if let Some(rb_palm) = rigid_body_set.get(palm_handle) {
                let colliders = rb_palm.colliders();
                if !colliders.is_empty() {
                    if let Some(collider) = collider_set.get(colliders[0]) {
                        let aabb = collider.compute_aabb();
                        println!("Palm boundary box: min=({:.3}, {:.3}), max=({:.3}, {:.3})\n",
                                 aabb.mins.x, aabb.mins.y, aabb.maxs.x, aabb.maxs.y);
                    }
                }
            }
        }
        
        // Apply a small force to the first box to create movement
        // if step < 50 {
        //     if let Some(rb1) = rigid_body_set.get_mut(tricep_handle) {
        //         rb1.add_force(vector![2.0, 0.0], true);
        //     }
        // }

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

        // Print boundary boxes every 20 steps
        
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