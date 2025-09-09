use rapier2d::dynamics::{CCDSolver, IntegrationParameters, IslandManager, RigidBodyBuilder};
use rapier2d::geometry::{ColliderBuilder, DefaultBroadPhase, NarrowPhase};
use rapier2d::na::{vector, Point2, Vector2};
use rapier2d::pipeline::PhysicsPipeline;
use rapier2d::prelude::nalgebra;
use crate::physics::{Corners};
use crate::physics::arm::{Arm, TRICEP_HALF_HEIGHT, TRICEP_MAX_FORCE};
use crate::physics::modelbody::{ModelBody, WorldSets};

// Ground dimensions
pub(super) const GROUND_HALF_WIDTH: f32 = 10.0;
pub(super) const GROUND_HALF_HEIGHT: f32 = 0.1;

const GROUND_MIDDLE_Y: f32 = -2.0;

// Wall dimensions
pub(super) const WALL_HALF_WIDTH: f32 = 0.3;
pub(super) const WALL_HALF_HEIGHT: f32 = 0.6;

pub(super) struct Hangman {
    pub(super) ground: ModelBody,
    pub(super) wall: ModelBody,
    pub(super) shoulder: ModelBody,
}

impl Hangman {
    pub fn new(world_sets: &mut WorldSets) -> Self {
        let ground_y = GROUND_MIDDLE_Y;
        let ground_top = ground_y + GROUND_HALF_HEIGHT;
        let ground = world_sets.create_body_with_builders(
            0.0, ground_y, RigidBodyBuilder::fixed(),
            GROUND_HALF_WIDTH, GROUND_HALF_HEIGHT, ColliderBuilder::cuboid(GROUND_HALF_WIDTH, GROUND_HALF_HEIGHT), 0.
        );

        // Create the wall sitting on top of the ground without overlap
        let wall_y = ground_top + WALL_HALF_HEIGHT;
        let wall = world_sets.create_body_with_builders(
            0.0, wall_y, RigidBodyBuilder::fixed(),
            WALL_HALF_WIDTH, WALL_HALF_HEIGHT, ColliderBuilder::cuboid(WALL_HALF_WIDTH, WALL_HALF_HEIGHT), 0.
        );

        let wall_far_side_centre = wall.get_far_side_centre(&world_sets.rigid_body_set);

        let shoulder = world_sets.create_body_with_builders(
            wall_far_side_centre.x, wall_far_side_centre.y, RigidBodyBuilder::fixed(),
            TRICEP_HALF_HEIGHT, TRICEP_HALF_HEIGHT, ColliderBuilder::ball(TRICEP_HALF_HEIGHT), TRICEP_MAX_FORCE
        );

        Self {
            ground,
            wall,
            shoulder,
        }
    }
}

pub struct PhysicsContext {
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    ccd_solver: CCDSolver,
    integration_parameters: IntegrationParameters,
    gravity: Vector2<f32>,
}

impl PhysicsContext {
    pub fn new() -> Self {
        let mut integration_parameters = IntegrationParameters::default();
        integration_parameters.dt = 1.0 / 250.0;
        integration_parameters.max_ccd_substeps = 16;
        Self {
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            ccd_solver: CCDSolver::new(),
            integration_parameters,
            gravity: vector![0.0, -9.81],
        }
    }

    pub(super) fn step(&mut self, world_sets: &mut WorldSets) {
        let physics_hooks = ();
        let event_handler = ();

        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut world_sets.rigid_body_set,
            &mut world_sets.collider_set,
            &mut world_sets.impulse_joint_set,
            &mut world_sets.multibody_joint_set,
            &mut self.ccd_solver,
            &physics_hooks,
            &event_handler,
        );
    }
}

pub struct PhysicsWorld {
    context: PhysicsContext,
    world_sets: WorldSets,
    arm: Arm,
    hangman: Hangman,
    ball: ModelBody,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        let mut world_sets = WorldSets::default();

        let hangman = Hangman::new(&mut world_sets);

        // Create the arm attached to the wall
        let arm = Arm::new(
            &mut world_sets,
            &hangman.shoulder,
        );
        let ground_top = hangman.ground.get_far_side_centre(&world_sets.rigid_body_set).y;

        // Create a pinchable ball positioned on the ground, about tricep length away from the wall
        let ball_radius = 0.03; // Small ball that can be pinched
        let ball_x = TRICEP_HALF_HEIGHT * 2.; // Position it away from the wall
        let ball_y = ground_top + ball_radius; // On the ground surface

        let ball = world_sets.create_dynamic_with_cb(
            ball_x, ball_y,ball_radius, ball_radius, ColliderBuilder::ball(ball_radius), 0.
        );

        Self {
            context: PhysicsContext::new(),
            arm,
            hangman,
            ball,
            world_sets,
        }
    }

    /// Steps the physics simulation forward by one frame
    pub fn step(&mut self) {
        self.context.step(&mut self.world_sets);
    }

    // Force application methods
    pub fn apply_tricep_force(&mut self, scaling_factor: f32) {
        self.arm
            .apply_tricep_force(&self.hangman.shoulder, scaling_factor, &mut self.world_sets.rigid_body_set)
    }

    pub fn apply_forearm_force(&mut self, scaling_factor: f32) {
        self.arm
            .apply_forearm_force(scaling_factor, &mut self.world_sets.rigid_body_set)
    }

    pub fn apply_palm_force(&mut self, scaling_factor: f32) {
        self.arm
            .apply_palm_force(scaling_factor, &mut self.world_sets.rigid_body_set)
    }

    pub fn apply_lower_index_finger_force(&mut self, scaling_factor: f32) {
        self.arm
            .apply_lower_index_finger_force(scaling_factor, &mut self.world_sets.rigid_body_set)
    }

    pub fn apply_upper_index_finger_force(&mut self, scaling_factor: f32) {
        self.arm
            .apply_upper_index_finger_force(scaling_factor, &mut self.world_sets.rigid_body_set)
    }

    pub fn apply_lower_thumb_force(&mut self, scaling_factor: f32) {
        self.arm
            .apply_lower_thumb_force(scaling_factor, &mut self.world_sets.rigid_body_set)
    }

    pub fn apply_upper_thumb_force(&mut self, scaling_factor: f32) {
        self.arm
            .apply_upper_thumb_force(scaling_factor, &mut self.world_sets.rigid_body_set)
    }

    // Farthest corners query methods
    pub fn tricep_farthest_corners(&self) -> Corners {
        self.arm
            .tricep_farthest_corners(&self.world_sets.rigid_body_set)
    }

    pub fn forearm_farthest_corners(&self) -> Corners {
        self.arm
            .forearm_farthest_corners(&self.world_sets.rigid_body_set)
    }

    pub fn palm_farthest_corners(&self) -> Corners {
        self.arm
            .palm_farthest_corners(&self.world_sets.rigid_body_set)
    }

    pub fn lower_index_finger_farthest_corners(&self) -> Corners {
        self.arm
            .lower_index_finger_farthest_corners(&self.world_sets.rigid_body_set)
    }

    pub fn upper_index_finger_farthest_corners(&self) -> Corners {
        self.arm
            .upper_index_finger_farthest_corners(&self.world_sets.rigid_body_set)
    }

    pub fn lower_thumb_farthest_corners(&self) -> Corners {
        self.arm
            .lower_thumb_farthest_corners(&self.world_sets.rigid_body_set)
    }

    pub fn upper_thumb_farthest_corners(&self) -> Corners {
        self.arm
            .upper_thumb_farthest_corners(&self.world_sets.rigid_body_set)
    }

    pub fn all_arm_corners(&self) -> Vec<[Point2<f32>; 4]> {
        self.arm
            .all_corners(&self.world_sets.rigid_body_set)
    }
}

#[cfg(test)]
mod tests {
    use crate::physics::world::PhysicsWorld;

    #[test]
    fn test_physics_simulation() {
        let mut world = PhysicsWorld::new();
        for i in 0..1000 {
            if i%10 == 0 {
                println!("{:?}", world.all_arm_corners());
            }
            world.apply_tricep_force(0.001417);
            // world.apply_forearm_force(-0.13);
            // world.apply_palm_force(-0.015);
            // world.apply_lower_index_finger_force(-0.03);
            world.step();
        }
    }

    #[test]
    fn test_force_locations() {
        let mut world = PhysicsWorld::new();
        let mut corners = world.all_arm_corners();
        corners.push(world.hangman.wall.get_bounding_box(&world.world_sets.rigid_body_set));
        corners.push(world.hangman.shoulder.get_bounding_box(&world.world_sets.rigid_body_set));
        println!("{:?}", corners);
        world.apply_tricep_force(0.001417);
        world.apply_forearm_force(-0.001417);
        world.apply_palm_force(0.001417);
        world.apply_lower_index_finger_force(-0.001417);
        world.apply_upper_index_finger_force(0.001417);
        world.apply_lower_thumb_force(-0.001417);
        world.apply_upper_thumb_force(0.001417);
    }

}