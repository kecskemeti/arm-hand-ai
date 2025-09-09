use std::sync::OnceLock;
use rapier2d::dynamics::{RigidBodySet};
use rapier2d::na::Point2;
use crate::physics::modelbody::{ModelBody, WorldSets};
use crate::physics::{Corners};
use crate::physics::modelbody::JoinType::{HorizontalJoin, VerticalJoin};

// Arm dimensions (half-extents!)
pub(super) const TRICEP_HALF_WIDTH: f32 = 0.155;
pub(super) const TRICEP_HALF_HEIGHT: f32 = 0.0375;

const FOREARM_HALF_WIDTH: f32 = 0.125;
const FOREARM_HALF_HEIGHT: f32 = 0.03;

const PALM_HALF_WIDTH: f32 = 0.05;
const PALM_HALF_HEIGHT: f32 = 0.01;

const FINGER_HALF_WIDTH: f32 = 0.0175;
const FINGER_HALF_HEIGHT: f32 = 0.008;

const THUMB_HALF_WIDTH: f32 = FINGER_HALF_HEIGHT;
const THUMB_HALF_HEIGHT: f32 = FINGER_HALF_WIDTH;

pub(super) const TRICEP_MAX_FORCE:f32 = 0.05;


pub(super) static X_RANGE:OnceLock<f32> = OnceLock::new();
pub(super) static Y_RANGE:OnceLock<f32> = OnceLock::new();
pub(super) static MIN_X:OnceLock<f32> = OnceLock::new();
pub(super) static MIN_Y:OnceLock<f32> = OnceLock::new();

pub(super) struct Arm {
    tricep_mb: ModelBody,
    forearm_mb: ModelBody,
    palm_mb: ModelBody,
    lower_index_finger_mb: ModelBody,
    upper_index_finger_mb: ModelBody,
    lower_thumb_mb: ModelBody,
    upper_thumb_mb: ModelBody,
}

impl Arm {
    pub fn new(
        world_sets: &mut WorldSets,
        shoulder_body: &ModelBody,
    ) -> Self {
        let shoulder_far_side_centre = shoulder_body.get_far_side_centre(&world_sets.rigid_body_set);

        // Calculate positions based on wall position and component dimensions
        let shoulder_right_edge = shoulder_far_side_centre.x;
        let shoulder_middle_y = shoulder_far_side_centre.y;
        // Tricep
        let tricep_mb = world_sets.create_joined_body_and_collider(
            &shoulder_body,
            HorizontalJoin,
            TRICEP_HALF_WIDTH,
            TRICEP_HALF_HEIGHT,
            TRICEP_MAX_FORCE,
        );

        // Forearm
        let forearm_mb = world_sets.create_joined_body_and_collider(&tricep_mb,
                                                                    HorizontalJoin,
                                                                    FOREARM_HALF_WIDTH,
                                                                    FOREARM_HALF_HEIGHT,
                                                                    TRICEP_MAX_FORCE/2.
        );

        // Palm
        let palm_mb = world_sets.create_joined_body_and_collider(&forearm_mb,
                                                                 HorizontalJoin,
                                                                 PALM_HALF_WIDTH,
                                                                 PALM_HALF_HEIGHT,
                                                                 TRICEP_MAX_FORCE/25.
        );

        // Lower index finger
        let lower_index_finger_mb = world_sets.create_joined_body_and_collider(&palm_mb,
                                                                               HorizontalJoin,
                                                                               FINGER_HALF_WIDTH,
                                                                               FINGER_HALF_HEIGHT,
                                                                               TRICEP_MAX_FORCE/40.
        );

        // Upper index finger
        let upper_index_finger_mb = world_sets.create_joined_body_and_collider(&lower_index_finger_mb,
                                                                               HorizontalJoin,
                                                                               FINGER_HALF_WIDTH,
                                                                               FINGER_HALF_HEIGHT,
                                                                               TRICEP_MAX_FORCE/50.
        );
        let farthest_point = upper_index_finger_mb.long_axis_farthest_corner(&world_sets.rigid_body_set);
        let _set_results = MIN_X.set(shoulder_right_edge - farthest_point.0.0)
            .and_then(|_| X_RANGE.set(farthest_point.0.0*2.))
            .and_then(|_| MIN_Y.set(shoulder_middle_y - farthest_point.0.0))
            .and_then(|_|Y_RANGE.set(farthest_point.0.0*2.));


        // Lower thumb
        let lower_thumb_mb = world_sets.create_joined_body_and_collider(&palm_mb,
                                                                        VerticalJoin,
                                                                        THUMB_HALF_WIDTH,
                                                                        THUMB_HALF_HEIGHT,
                                                                        TRICEP_MAX_FORCE/40.
        );

        // Upper thumb
        let upper_thumb_mb = world_sets.create_joined_body_and_collider(&lower_thumb_mb,
                                                                        VerticalJoin,
                                                                        THUMB_HALF_WIDTH,
                                                                        THUMB_HALF_HEIGHT,
                                                                        TRICEP_MAX_FORCE/50.
        );

        Self {
            tricep_mb,
            forearm_mb,
            palm_mb,
            lower_index_finger_mb,
            upper_index_finger_mb,
            lower_thumb_mb,
            upper_thumb_mb,
        }
    }

    pub fn all_corners(
        &self,
        rigid_body_set: &RigidBodySet,
    ) -> Vec<[Point2<f32>; 4]> {
        [
            self.tricep_mb,
            self.forearm_mb,
            self.palm_mb,
            self.lower_index_finger_mb,
            self.upper_index_finger_mb,
            self.lower_thumb_mb,
            self.upper_thumb_mb,
        ]
            .iter()
            .map(|&rb_handle| rb_handle.get_bounding_box(rigid_body_set))
            .collect()
    }

    pub fn tricep_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
    ) -> Corners {
        self.tricep_mb.long_axis_farthest_corner(rigid_body_set)
    }

    pub fn forearm_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
    ) -> Corners {
        self.forearm_mb.long_axis_farthest_corner(rigid_body_set)
    }

    pub fn palm_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
    ) -> Corners {
        self.palm_mb.long_axis_farthest_corner(rigid_body_set)
    }

    pub fn lower_index_finger_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
    ) -> Corners {
        self.lower_index_finger_mb.long_axis_farthest_corner(rigid_body_set)
    }

    pub fn upper_index_finger_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
    ) -> Corners {
        self.upper_index_finger_mb.long_axis_farthest_corner(rigid_body_set)
    }

    pub fn lower_thumb_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
    ) -> Corners {
        self.lower_thumb_mb.long_axis_farthest_corner(rigid_body_set)
    }

    pub fn upper_thumb_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
    ) -> Corners {
        self.upper_thumb_mb.long_axis_farthest_corner(rigid_body_set)
    }

    pub fn apply_tricep_force(
        &self,
        shoulder: &ModelBody,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) {
        ModelBody::apply_force_between(shoulder, &self.tricep_mb, rigid_body_set, scaling_factor);
    }

    pub fn apply_forearm_force(
        &self,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) {
        ModelBody::apply_force_between(&self.tricep_mb, &self.forearm_mb, rigid_body_set, scaling_factor);
    }

    pub fn apply_palm_force(&self, scaling_factor: f32, rigid_body_set: &mut RigidBodySet) {
        ModelBody::apply_force_between(&self.forearm_mb, &self.palm_mb, rigid_body_set, scaling_factor);
    }

    pub fn apply_lower_index_finger_force(
        &self,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) {
        ModelBody::apply_force_between(&self.palm_mb, &self.lower_index_finger_mb, rigid_body_set, scaling_factor);
    }

    pub fn apply_upper_index_finger_force(
        &self,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) {
        ModelBody::apply_force_between(&self.lower_index_finger_mb, &self.upper_index_finger_mb, rigid_body_set, scaling_factor);
    }

    pub fn apply_lower_thumb_force(
        &self,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) {
        ModelBody::apply_force_between(&self.palm_mb, &self.lower_thumb_mb, rigid_body_set, scaling_factor);
    }

    pub fn apply_upper_thumb_force(
        &self,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) {
        ModelBody::apply_force_between(&self.lower_thumb_mb, &self.upper_thumb_mb, rigid_body_set, scaling_factor);
    }
}

pub fn normalize_x(x_value: f32) -> f32 {
    (x_value - MIN_X.get().unwrap()) / X_RANGE.get().unwrap()
}

pub fn normalize_y(y_value: f32) -> f32 {
    (y_value - MIN_Y.get().unwrap()) / Y_RANGE.get().unwrap()
}

#[cfg(test)]
mod tests {
    use rapier2d::na::distance;
    use crate::physics::world::Hangman;
    use super::*;

    #[test]
    pub fn test_arm() {
        let mut world = WorldSets::default();
        let hangman = Hangman::new(&mut world);
        let arm = Arm::new(&mut world, &hangman.shoulder);
        let corners = arm.all_corners(&world.rigid_body_set);
        let expectations = [
            (0,1,TRICEP_HALF_WIDTH*2.),
            (0,1, FOREARM_HALF_WIDTH*2.),
            (0,1,PALM_HALF_WIDTH*2.),
            (0,1, FINGER_HALF_WIDTH*2.),
            (0,1, FINGER_HALF_WIDTH*2.),
            (1,2, FOREARM_HALF_WIDTH*2.),
            (1,2, FOREARM_HALF_WIDTH*2.),
        ];
        for (corner, expectation) in corners.iter().zip(expectations.iter()) {
            assert!(distance(&corner[expectation.0], &corner[expectation.1])-expectation.2<0.0001);
        }
    }
}