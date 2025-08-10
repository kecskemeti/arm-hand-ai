use rapier2d::dynamics::{
    CCDSolver, ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet,
    RevoluteJointBuilder, RigidBodyBuilder, RigidBodyHandle, RigidBodySet, RigidBodyType,
};
use rapier2d::geometry::{ColliderBuilder, ColliderSet, DefaultBroadPhase, NarrowPhase};
use rapier2d::na::{distance, point, vector, Point2, Vector2};
use rapier2d::pipeline::{ActiveEvents, PhysicsPipeline};
use rapier2d::prelude::nalgebra;
use crate::physics::JoinType::{HorizontalJoin, VerticalJoin};

pub type Corners=((f32,f32),(f32,f32));

// Ground dimensions
const GROUND_HALF_WIDTH: f32 = 100.0;
const GROUND_HALF_HEIGHT: f32 = 0.1;

// Wall dimensions
const WALL_HALF_WIDTH: f32 = 0.2;
const WALL_HALF_HEIGHT: f32 = 2.0;

// Cuboid dimensions (half-extents)
const TRICEP_HALF_WIDTH: f32 = PALM_HALF_WIDTH * 3.;
const TRICEP_HALF_HEIGHT: f32 = 0.04;

const FOREARM_HALF_WIDTH: f32 = PALM_HALF_WIDTH*2.5;
const FOREARM_HALF_HEIGHT: f32 = 0.035;

const PALM_HALF_WIDTH: f32 = 0.05;
const PALM_HALF_HEIGHT: f32 = 0.0175;

const FINGER_HALF_WIDTH: f32 = PALM_HALF_WIDTH/1.33/2.;
const FINGER_HALF_HEIGHT: f32 = 0.01;

const THUMB_HALF_WIDTH: f32 = FINGER_HALF_HEIGHT;
const THUMB_HALF_HEIGHT: f32 = FINGER_HALF_WIDTH;

// Joint anchor points (relative to cuboid centers)
const WALL_SHOULDER_ANCHOR: Point2<f32> = point![WALL_HALF_WIDTH, 0.0];
const TRICEP_SHOULDER_ANCHOR: Point2<f32> = point![-TRICEP_HALF_WIDTH, 0.0]; // At left edge
const TRICEP_ELBOW_ANCHOR: Point2<f32> = point![TRICEP_HALF_WIDTH * 2.0, 0.0]; // At right edge
const FOREARM_WRIST_ANCHOR: Point2<f32> = point![FOREARM_HALF_WIDTH * 1.6, 0.0]; // Slightly inward from right edge
const PALM_INDEX_ANCHOR: Point2<f32> = point![PALM_HALF_WIDTH, 0.0]; // At right edge
const PALM_THUMB_ANCHOR: Point2<f32> = point![0.0, -PALM_HALF_HEIGHT]; // At bottom edge

const FINGER_JOINT_ANCHOR: Point2<f32> = point![FINGER_HALF_WIDTH, 0.0]; // At right edge

const THUMB_JOINT_ANCHOR_BOTTOM: Point2<f32> = point![0.0, -THUMB_HALF_HEIGHT]; // Near bottom

const MAX_X: f32 = WALL_HALF_WIDTH
    + TRICEP_HALF_WIDTH * 2.0
    + FOREARM_HALF_WIDTH * 2.0
    + PALM_HALF_WIDTH * 2.0
    + FINGER_HALF_WIDTH * 2.0
    + FINGER_HALF_WIDTH * 2.0;

pub const MIN_X: f32 = WALL_HALF_WIDTH;

const MAX_Y: f32 = WALL_HALF_HEIGHT + MAX_X;

const MIN_Y: f32 = GROUND_MIDDLE_Y + GROUND_HALF_HEIGHT;

const GROUND_MIDDLE_Y: f32 = -2.0;

const X_RANGE: f32 = MAX_X - MIN_X;
const Y_RANGE: f32 = MAX_Y - MIN_Y;

fn create_dynamic_body(
    body_set: &mut RigidBodySet,
    centre_x: f32,
    centre_y: f32,
) -> RigidBodyHandle {
    body_set.insert(
        RigidBodyBuilder::dynamic()
            .translation(vector![centre_x, centre_y])
            .can_sleep(false)
            .ccd_enabled(true)
            .build(),
    )
}

fn joint_between_rigid_bodies(
    rb1: RigidBodyHandle,
    point1: Point2<f32>,
    rb2: RigidBodyHandle,
    point2: Point2<f32>,
    joint_set: &mut ImpulseJointSet,
) {
    let joint = RevoluteJointBuilder::new()
        .local_anchor1(point1)
        .local_anchor2(point2)
        .build();

    joint_set.insert(rb1, rb2, joint, true);
}

pub struct ModelBodyBuilder<'a> {
    rigid_body_set: &'a mut RigidBodySet,
    collider_set: &'a mut ColliderSet,
    impulse_joint_set: &'a mut ImpulseJointSet,
}

impl ModelBodyBuilder<'_> {
    fn create_joined_body_and_collider(&mut self,
                                       root: &ModelBody,
                                       join: JoinType,
                                       centre_x: f32,
                                       centre_y: f32,
                                       width: f32,
                                       height: f32,
    ) -> ModelBody {
        root.create_joined_body_and_collider(
            join,
            self.rigid_body_set,
            centre_x,
            centre_y,
            self.collider_set,
            width,
            height,
            self.impulse_joint_set
        )
    }

    fn create_body_and_collider(
        &mut self,
        centre_x: f32,
        centre_y: f32,
        width: f32,
        height: f32,
    ) -> ModelBody {
        ModelBody::create_body_and_collider(
            self.rigid_body_set,
            centre_x,
            centre_y,
            self.collider_set,
            width,
            height,
        )
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum JoinType {
    HorizontalJoin,
    VerticalJoin,
}

#[derive(Copy, Clone)]
pub struct ModelBody {
    rb: RigidBodyHandle,
    bounding_box: [Point2<f32>; 4],
}

impl ModelBody {
    pub fn get_bounding_box(&self, rigid_body_set: &RigidBodySet) -> [Point2<f32>; 4] {
        let body = &rigid_body_set[self.rb].position();
        self.bounding_box.iter().map(|p| *body * *p).collect::<Vec<_>>().try_into().unwrap()
    }

    fn create_box_and_collider(
        body_set: &mut RigidBodySet,
        centre_x: f32,
        centre_y: f32,
        collider_set: &mut ColliderSet,
        width: f32,
        height: f32,
        cb: ColliderBuilder
    ) -> Self {
        let body_handle = create_dynamic_body(body_set, centre_x, centre_y);
        let collider_handle = cb
            .restitution(0.7)
            .friction(0.3)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .build();
        collider_set.insert_with_parent(collider_handle, body_handle, body_set);
        Self {
            rb: body_handle,
            bounding_box: [
                point!(-width, height),
                point!(width, height),
                point!(width, -height),
                point!(-width, -height),
            ]
        }
    }

    fn create_body_and_collider(
        body_set: &mut RigidBodySet,
        centre_x: f32,
        centre_y: f32,
        collider_set: &mut ColliderSet,
        width: f32,
        height: f32,
    ) -> Self {
        Self::create_box_and_collider(body_set,centre_x,centre_y,collider_set,width,height, if width>height {
            ColliderBuilder::capsule_x(width, height)
        } else {
            ColliderBuilder::capsule_y(height, width)
        })
    }

    fn create_joined_body_and_collider(
        &self,
        join: JoinType,
        body_set: &mut RigidBodySet,
        centre_x: f32,
        centre_y: f32,
        collider_set: &mut ColliderSet,
        width: f32,
        height: f32, impulse_joint_set: &mut ImpulseJointSet
    ) -> Self {
        let follower = Self::create_body_and_collider(body_set, centre_x, centre_y, collider_set, width, height);
        if join == HorizontalJoin {
            self.join_horizontal_rigid_bodies(&follower, impulse_joint_set)
        } else {
            self.join_vertical_rigid_bodies(&follower, impulse_joint_set)
        }
        follower
    }

    fn join_horizontal_rigid_bodies(
        &self,
        other: &Self,
        joint_set: &mut ImpulseJointSet,
    ) {
        self.join_with_anchors(other, joint_set, point![self.bounding_box[1].x, 0.0], point![other.bounding_box[0].x, 0.0])
    }

    fn join_vertical_rigid_bodies(&self, other: &Self, joint_set:&mut ImpulseJointSet) {
        self.join_with_anchors(other, joint_set, point![0.0, self.bounding_box[2].y], point![0.0, other.bounding_box[1].y])
    }

    fn join_with_anchors(&self, other:&Self, joint_set: &mut ImpulseJointSet, self_anchor:Point2<f32>, other_anchor:Point2<f32>) {
        let joint = RevoluteJointBuilder::new()
            .local_anchor1(self_anchor)
            .local_anchor2(other_anchor)
            .build();

        joint_set.insert(self.rb, other.rb, joint, true);
    }

    fn long_axis_farthest_corner(&self, rigid_body_set: &RigidBodySet) -> Corners {
        let bb = self.get_bounding_box(rigid_body_set);
        if distance(&bb[0],&bb[1])> distance(&bb[1], &bb[2]) {
            ((bb[1].x, bb[1].y), (bb[2].x, bb[2].y))
        } else {
            ((bb[2].x, bb[2].y), (bb[3].x, bb[3].y))
        }
    }

}

pub struct Arm {
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
        rigid_body_set: &mut RigidBodySet,
        collider_set: &mut ColliderSet,
        impulse_joint_set: &mut ImpulseJointSet,
        wall_handle: RigidBodyHandle,
    ) -> Self {
        let wall_rb = rigid_body_set.get(wall_handle).expect("Wall not found.");
        let wall_middle_y = wall_rb.translation().y;
        let wall_x = wall_rb.translation().x;

        // Calculate positions based on wall position and component dimensions
        let wall_right_edge = wall_x + WALL_HALF_WIDTH;
        let tricep_x = wall_right_edge + TRICEP_HALF_WIDTH;
        let forearm_x =
            tricep_x + TRICEP_HALF_WIDTH + FOREARM_HALF_WIDTH;
        let palm_x = forearm_x + FOREARM_HALF_WIDTH + PALM_HALF_WIDTH;
        let lower_finger_x = palm_x + PALM_HALF_WIDTH + FINGER_HALF_WIDTH;
        let upper_finger_x =
            lower_finger_x + FINGER_HALF_WIDTH + FINGER_HALF_WIDTH;
        let mut mb_builder = ModelBodyBuilder {
            impulse_joint_set, collider_set, rigid_body_set
        };

        // Tricep
        let tricep_mb = mb_builder.create_body_and_collider(
            tricep_x,
            wall_middle_y,
            TRICEP_HALF_WIDTH,
            TRICEP_HALF_HEIGHT,
        );
        joint_between_rigid_bodies(
            wall_handle,
            WALL_SHOULDER_ANCHOR,
            tricep_mb.rb,
            TRICEP_SHOULDER_ANCHOR,
            mb_builder.impulse_joint_set,
        );

        // Forearm
        let forearm_mb = mb_builder.create_joined_body_and_collider(&tricep_mb,
                                                                    HorizontalJoin,
                                                                    forearm_x,
                                                                    wall_middle_y,
                                                                    FOREARM_HALF_WIDTH,
                                                                    FOREARM_HALF_HEIGHT,
        );

        // Palm
        let palm_mb = mb_builder.create_joined_body_and_collider(&forearm_mb,
                                                                 HorizontalJoin,
                                                                 palm_x,
                                                                 wall_middle_y,
                                                                 PALM_HALF_WIDTH,
                                                                 PALM_HALF_HEIGHT,
        );

        // Lower index finger
        let lower_index_finger_mb = mb_builder.create_joined_body_and_collider(&palm_mb,
                                                                               HorizontalJoin,
                                                                               lower_finger_x,
                                                                               wall_middle_y,
                                                                               FINGER_HALF_WIDTH,
                                                                               FINGER_HALF_HEIGHT,
        );

        // Upper index finger
        let upper_index_finger_mb = mb_builder.create_joined_body_and_collider(&lower_index_finger_mb,
                                                                               HorizontalJoin,
                                                                               upper_finger_x,
                                                                               wall_middle_y,
                                                                               FINGER_HALF_WIDTH,
                                                                               FINGER_HALF_HEIGHT,
        );

        // Lower thumb
        let lower_thumb_mb = mb_builder.create_joined_body_and_collider(&palm_mb,
                                                                        VerticalJoin,
                                                                        palm_x,
                                                                        wall_middle_y-THUMB_HALF_HEIGHT-PALM_HALF_HEIGHT,
                                                                        THUMB_HALF_WIDTH,
                                                                        THUMB_HALF_HEIGHT,
        );

        // Upper thumb
        let upper_thumb_mb = mb_builder.create_joined_body_and_collider(&lower_thumb_mb,
                                                                        VerticalJoin,
                                                                        palm_x,
                                                                        wall_middle_y - THUMB_HALF_HEIGHT*3.-PALM_HALF_HEIGHT,
                                                                        THUMB_HALF_WIDTH,
                                                                        THUMB_HALF_HEIGHT,
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

    pub fn print_state(&self, rigid_body_set: &RigidBodySet, collider_set: &ColliderSet) {
        let handles = [
            ("Tricep", self.tricep_mb),
            ("Forearm", self.forearm_mb),
            ("Palm", self.palm_mb),
            ("Lower Index Finger", self.lower_index_finger_mb),
            ("Upper Index Finger", self.upper_index_finger_mb),
            ("Lower Thumb", self.lower_thumb_mb),
            ("Upper Thumb", self.upper_thumb_mb),
        ];
        for (name, handle) in handles {
            if let Some(rb) = rigid_body_set.get(handle.rb) {
                let colliders = rb.colliders();
                if colliders.is_empty() {
                    println!("{} has no attached colliders.", name);
                    continue;
                }
                for &collider_handle in colliders {
                    if let Some(collider) = collider_set.get(collider_handle) {
                        let aabb = collider.compute_aabb();
                        println!(
                            "{} boundary box: min=({:.3}, {:.3}), max=({:.3}, {:.3})",
                            name, aabb.mins.x, aabb.mins.y, aabb.maxs.x, aabb.maxs.y
                        );
                    }
                }
            } else {
                println!("{} rigid body not found.", name);
            }
        }
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

    /// Applies a scaled force to a specified rigid body, pointing toward or away from an adjusted position relative to a joint.
    /// All previous forces on the target rigid body are cleared first.
    ///
    /// # Arguments
    /// * `target_handle` - Handle of the rigid body to apply force to
    /// * `joint_handle` - Handle of the rigid body containing the reference joint
    /// * `joint_anchor` - Local anchor point on the joint rigid body
    /// * `adjustment` - Offset vector to add to the joint position (e.g., vector![0.0, 0.05] for 0.05 units above)
    /// * `max_force_magnitude` - Maximum force magnitude when scaling_factor is ±1.0
    /// * `scaling_factor` - A value between -1.0 and 1.0:
    ///   - -1.0: Maximum force toward the adjusted joint position (attraction)
    ///   - 1.0: Maximum force away from the adjusted joint position (repulsion)
    ///   - 0.0: No force applied
    /// * `rigid_body_set` - Mutable reference to the rigid body set
    ///
    /// # Returns
    /// * `true` if the force was successfully applied
    /// * `false` if either rigid body could not be found
    fn apply_force_to_body(
        &self,
        target_handle: RigidBodyHandle,
        joint_handle: RigidBodyHandle,
        joint_anchor: Point2<f32>,
        adjustment: Vector2<f32>,
        max_force_magnitude: f32,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) -> bool {
        // Clamp scaling factor to valid range
        let scaling_factor = scaling_factor.clamp(-1.0, 1.0);

        // Get the target rigid body
        let target_rb = match rigid_body_set.get_mut(target_handle) {
            Some(rb) => rb,
            None => return false,
        };

        // Clear all existing forces on the target body
        target_rb.reset_forces(true);

        // If scaling factor is 0, no force to apply
        if scaling_factor.abs() < f32::EPSILON {
            return true;
        }

        // Get the joint position from the joint rigid body
        let joint_pos = if let Some(joint_rb) = rigid_body_set.get(joint_handle) {
            let joint_rb_pos = joint_rb.position();
            joint_rb_pos.rotation.transform_point(&joint_anchor) + joint_rb_pos.translation.vector
        } else {
            return false;
        };

        // Calculate the target position (joint position + adjustment)
        let target_pos = Point2::new(joint_pos.x + adjustment.x, joint_pos.y + adjustment.y);

        // Get the target body's center of mass position
        let target_body_pos = rigid_body_set
            .get(target_handle)
            .expect("force target not found")
            .center_of_mass();

        // Calculate the direction vector from target body to target position
        let direction = target_pos - target_body_pos;
        let direction_norm = direction.norm();

        // Avoid division by zero
        if direction_norm < f32::EPSILON {
            return true;
        }

        // Normalize the direction vector
        let direction_unit = direction / direction_norm;

        // Apply scaling factor to get actual force magnitude
        let force_magnitude = max_force_magnitude * scaling_factor;

        // Calculate final force vector
        let force_vector = if scaling_factor < 0.0 {
            // Attraction: force toward target position
            direction_unit * force_magnitude.abs()
        } else {
            // Repulsion: force away from target position
            -direction_unit * force_magnitude
        };

        // Apply the force to the target body
        let target_rb = rigid_body_set
            .get_mut(target_handle)
            .expect("by this time we failed already when we query the target position");
        target_rb.add_force(force_vector, true);

        true
    }

    /// Applies a scaled force to the tricep, pointing toward or away from a position 0.05 units above the wall joint.
    pub fn apply_tricep_force(
        &self,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) -> bool {
        // Find the wall rigid body handle
        let wall_handle = if let Some((handle, _)) = rigid_body_set
            .iter()
            .find(|(_, rb)| rb.body_type() == RigidBodyType::Fixed)
        {
            handle
        } else {
            return false;
        };

        self.apply_force_to_body(
            self.tricep_mb.rb,
            wall_handle,
            WALL_SHOULDER_ANCHOR,
            vector![0.0, 0.05],
            5.0,
            scaling_factor,
            rigid_body_set,
        )
    }

    /// Applies a scaled force to the forearm, pointing toward or away from a position 0.05 units above the elbow joint.
    pub fn apply_forearm_force(
        &self,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) -> bool {
        self.apply_force_to_body(
            self.forearm_mb.rb,
            self.tricep_mb.rb,
            TRICEP_ELBOW_ANCHOR,
            vector![0.0, 0.05],
            2.5,
            scaling_factor,
            rigid_body_set,
        )
    }

    /// Applies a scaled force to the palm, pointing toward or away from a position 0.05 units above the wrist joint.
    pub fn apply_palm_force(&self, scaling_factor: f32, rigid_body_set: &mut RigidBodySet) -> bool {
        self.apply_force_to_body(
            self.palm_mb.rb,
            self.forearm_mb.rb,
            FOREARM_WRIST_ANCHOR,
            vector![0.0, 0.05],
            2.0, // Smaller force for palm
            scaling_factor,
            rigid_body_set,
        )
    }

    /// Applies a scaled force to the lower index finger, pointing toward or away from a position 0.05 units above the palm joint.
    pub fn apply_lower_index_finger_force(
        &self,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) -> bool {
        self.apply_force_to_body(
            self.lower_index_finger_mb.rb,
            self.palm_mb.rb,
            PALM_INDEX_ANCHOR,
            vector![0.0, 0.05],
            1.5, // Smaller force for finger segments
            scaling_factor,
            rigid_body_set,
        )
    }

    /// Applies a scaled force to the upper index finger, pointing toward or away from a position 0.05 units above the middle finger joint.
    pub fn apply_upper_index_finger_force(
        &self,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) -> bool {
        self.apply_force_to_body(
            self.upper_index_finger_mb.rb,
            self.lower_index_finger_mb.rb,
            FINGER_JOINT_ANCHOR,
            vector![0.0, 0.05],
            1.0, // Smallest force for fingertip
            scaling_factor,
            rigid_body_set,
        )
    }

    /// Applies a scaled force to the lower thumb, pointing toward or away from a position 0.05 units above the palm-thumb joint.
    pub fn apply_lower_thumb_force(
        &self,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) -> bool {
        self.apply_force_to_body(
            self.lower_thumb_mb.rb,
            self.palm_mb.rb,
            PALM_THUMB_ANCHOR,
            vector![0.0, 0.05],
            1.5, // Same as finger segments
            scaling_factor,
            rigid_body_set,
        )
    }

    /// Applies a scaled force to the upper thumb, pointing toward or away from a position 0.05 units above the middle thumb joint.
    pub fn apply_upper_thumb_force(
        &self,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) -> bool {
        self.apply_force_to_body(
            self.upper_thumb_mb.rb,
            self.lower_thumb_mb.rb,
            THUMB_JOINT_ANCHOR_BOTTOM,
            vector![0.0, 0.05],
            1.0, // Smallest force for thumb tip
            scaling_factor,
            rigid_body_set,
        )
    }

    /// Gets the handle for the upper thumb segment
    pub fn upper_thumb_handle(&self) -> RigidBodyHandle {
        self.upper_thumb_mb.rb
    }

    /// Gets the handle for the upper index finger segment
    pub fn upper_index_finger_handle(&self) -> RigidBodyHandle {
        self.upper_index_finger_mb.rb
    }

    /// Gets handles for all arm segments
    pub fn all_handles(&self) -> [RigidBodyHandle; 7] {
        [
            self.tricep_mb.rb,
            self.forearm_mb.rb,
            self.palm_mb.rb,
            self.lower_index_finger_mb.rb,
            self.upper_index_finger_mb.rb,
            self.lower_thumb_mb.rb,
            self.upper_thumb_mb.rb,
        ]
    }
}

pub struct PhysicsWorld {
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    ccd_solver: CCDSolver,
    integration_parameters: IntegrationParameters,
    gravity: Vector2<f32>,
    arm: Arm,
    _wall_handle: RigidBodyHandle,
    _ground_handle: RigidBodyHandle,
    ball_handle: RigidBodyHandle,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();
        let mut impulse_joint_set = ImpulseJointSet::new();

        // Create the ground
        let ground_y = GROUND_MIDDLE_Y;
        let ground_rigid_body = RigidBodyBuilder::fixed()
            .translation(vector![0.0, ground_y])
            .build();
        let ground_handle = rigid_body_set.insert(ground_rigid_body);

        let ground_collider = ColliderBuilder::cuboid(GROUND_HALF_WIDTH, GROUND_HALF_HEIGHT)
            .restitution(0.7)
            .friction(0.3)
            .build();
        collider_set.insert_with_parent(ground_collider, ground_handle, &mut rigid_body_set);

        // Create the wall sitting on top of the ground without overlap
        let ground_top = ground_y + GROUND_HALF_HEIGHT;
        let wall_y = ground_top + WALL_HALF_HEIGHT;
        let wall_rigid_body = RigidBodyBuilder::fixed()
            .translation(vector![0.0, wall_y])
            .build();
        let wall_handle = rigid_body_set.insert(wall_rigid_body);

        let wall_collider = ColliderBuilder::cuboid(WALL_HALF_WIDTH, WALL_HALF_HEIGHT)
            .restitution(0.7)
            .friction(0.3)
            .build();
        collider_set.insert_with_parent(wall_collider, wall_handle, &mut rigid_body_set);

        // Create the arm attached to the wall
        let arm = Arm::new(
            &mut rigid_body_set,
            &mut collider_set,
            &mut impulse_joint_set,
            wall_handle,
        );

        // Create a pinchable ball positioned on the ground, about tricep length away from the wall
        let ball_radius = 0.03; // Small ball that can be pinched
        let ball_x = TRICEP_HALF_WIDTH * 2.; // Position it away from the wall
        let ball_y = ground_top + ball_radius; // On the ground surface

        let ball_rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![ball_x, ball_y])
            .build();
        let ball_handle = rigid_body_set.insert(ball_rigid_body);

        let ball_collider = ColliderBuilder::ball(ball_radius)
            .restitution(0.3)
            .friction(0.8) // Higher friction to make it easier to grip
            .density(0.5) // Light ball
            .build();
        collider_set.insert_with_parent(ball_collider, ball_handle, &mut rigid_body_set);

        // Set up physics parameters
        let gravity = vector![0.0, -9.81];
        let mut integration_parameters = IntegrationParameters::default();
        integration_parameters.dt = 1.0 / 240.0;
        integration_parameters.max_ccd_substeps = 4;

        Self {
            rigid_body_set,
            collider_set,
            impulse_joint_set,
            multibody_joint_set: MultibodyJointSet::new(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            ccd_solver: CCDSolver::new(),
            integration_parameters,
            gravity,
            arm,
            _wall_handle: wall_handle,
            _ground_handle: ground_handle,
            ball_handle,
        }
    }

    /// Steps the physics simulation forward by one frame
    pub fn step(&mut self) {
        let physics_hooks = ();
        let event_handler = ();

        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            &physics_hooks,
            &event_handler,
        );
    }

    /// Prints the current state of all arm components
    pub fn print_arm_state(&self) {
        self.arm
            .print_state(&self.rigid_body_set, &self.collider_set);
    }

    // Force application methods
    pub fn apply_tricep_force(&mut self, scaling_factor: f32) -> bool {
        self.arm
            .apply_tricep_force(scaling_factor, &mut self.rigid_body_set)
    }

    pub fn apply_forearm_force(&mut self, scaling_factor: f32) -> bool {
        self.arm
            .apply_forearm_force(scaling_factor, &mut self.rigid_body_set)
    }

    pub fn apply_palm_force(&mut self, scaling_factor: f32) -> bool {
        self.arm
            .apply_palm_force(scaling_factor, &mut self.rigid_body_set)
    }

    pub fn apply_lower_index_finger_force(&mut self, scaling_factor: f32) -> bool {
        self.arm
            .apply_lower_index_finger_force(scaling_factor, &mut self.rigid_body_set)
    }

    pub fn apply_upper_index_finger_force(&mut self, scaling_factor: f32) -> bool {
        self.arm
            .apply_upper_index_finger_force(scaling_factor, &mut self.rigid_body_set)
    }

    pub fn apply_lower_thumb_force(&mut self, scaling_factor: f32) -> bool {
        self.arm
            .apply_lower_thumb_force(scaling_factor, &mut self.rigid_body_set)
    }

    pub fn apply_upper_thumb_force(&mut self, scaling_factor: f32) -> bool {
        self.arm
            .apply_upper_thumb_force(scaling_factor, &mut self.rigid_body_set)
    }

    // Farthest corners query methods
    pub fn tricep_farthest_corners(&self) -> Corners {
        self.arm
            .tricep_farthest_corners(&self.rigid_body_set)
    }

    pub fn forearm_farthest_corners(&self) -> Corners {
        self.arm
            .forearm_farthest_corners(&self.rigid_body_set)
    }

    pub fn palm_farthest_corners(&self) -> Corners {
        self.arm
            .palm_farthest_corners(&self.rigid_body_set)
    }

    pub fn lower_index_finger_farthest_corners(&self) -> Corners {
        self.arm
            .lower_index_finger_farthest_corners(&self.rigid_body_set)
    }

    pub fn upper_index_finger_farthest_corners(&self) -> Corners {
        self.arm
            .upper_index_finger_farthest_corners(&self.rigid_body_set)
    }

    pub fn lower_thumb_farthest_corners(&self) -> Corners {
        self.arm
            .lower_thumb_farthest_corners(&self.rigid_body_set)
    }

    pub fn upper_thumb_farthest_corners(&self) -> Corners {
        self.arm
            .upper_thumb_farthest_corners(&self.rigid_body_set)
    }

    pub fn all_arm_corners(&self) -> Vec<[Point2<f32>; 4]> {
        self.arm
            .all_corners(&self.rigid_body_set)
    }

    /// Gets the ball's position in world coordinates
    pub fn ball_position(&self) -> Option<(f32, f32)> {
        let ball_rb = self.rigid_body_set.get(self.ball_handle)?;
        let translation = ball_rb.translation();
        Some((translation.x, translation.y))
    }

    /// Prints ball state information
    pub fn print_ball_state(&self) {
        if let Some((x, y)) = self.ball_position() {
            println!("Ball: pos=({:.3}, {:.3})", x, y);
        }
    }
}

pub fn normalize_x(x_value: f32) -> f32 {
    (x_value - MIN_X) / X_RANGE
}

pub fn normalize_y(y_value: f32) -> f32 {
    (y_value - MIN_Y) / Y_RANGE
}

#[cfg(test)]
mod tests {
    use crate::physics::{joint_between_rigid_bodies, ModelBody, PhysicsWorld};
    use rapier2d::dynamics::{
        CCDSolver, IntegrationParameters, IslandManager, RigidBodyBuilder, RigidBodySet,
    };
    use rapier2d::geometry::{ColliderBuilder, ColliderSet};
    use rapier2d::na::{distance, point, vector};
    use rapier2d::prelude::{nalgebra, MultibodyJointSet, NarrowPhase, PhysicsPipeline};
    use rapier2d::prelude::{DefaultBroadPhase, ImpulseJointSet};

    #[test]
    fn test_physics_simulation() {
        let mut world = PhysicsWorld::new();
        for i in 0..1000 {
            if i%10 == 0 {
                println!("{:?}", world.all_arm_corners());
            }
            world.apply_tricep_force(0.023);
            world.apply_forearm_force(-0.13);
            world.apply_palm_force(-0.015);
            world.apply_lower_index_finger_force(-0.03);
            world.step();
        }
    }

    #[test]
    fn single_body_define() {
        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();
        let centre_x = 0.47;
        let centre_y = 0.1;
        let half_width = 0.25;
        let half_height = 0.04;
        let body_mb = ModelBody::create_body_and_collider(
            &mut rigid_body_set,
            centre_x,
            centre_y,
            &mut collider_set,
            half_width,
            half_height,
        );
        let body_pos = rigid_body_set[body_mb.rb].position().translation;
        assert_eq!(body_pos.x, centre_x);
        assert_eq!(body_pos.y, centre_y);
        let collider_pos = body_mb.get_bounding_box(&rigid_body_set);
        assert_eq!(collider_pos[0].x, centre_x - half_width);
        assert_eq!(collider_pos[0].y, centre_y + half_height);
        assert_eq!(collider_pos[1].x, centre_x + half_width);
        assert_eq!(collider_pos[1].y, centre_y + half_height);
        assert_eq!(collider_pos[2].x, centre_x + half_width);
        assert_eq!(collider_pos[2].y, centre_y - half_height);
        assert_eq!(collider_pos[3].x, centre_x - half_width);
        assert_eq!(collider_pos[3].y, centre_y - half_height);
    }

    #[test]
    fn bodyies_with_joint_define() {
        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();
        let mut impulse_joint_set = ImpulseJointSet::new();
        let centre_x = 0.45;
        let centre_y = 0.1;
        let half_width = 0.25;
        let half_height = 0.04;
        let wall_width = 0.2;
        let wall_rigid_body = RigidBodyBuilder::fixed()
            .translation(vector![0.0, 0.1])
            .build();
        let wall_handle = rigid_body_set.insert(wall_rigid_body);

        let wall_collider = ColliderBuilder::cuboid(wall_width, 2.0)
            .restitution(0.7)
            .friction(0.3)
            .build();
        collider_set.insert_with_parent(wall_collider, wall_handle, &mut rigid_body_set);

        let body_mb = ModelBody::create_body_and_collider(
            &mut rigid_body_set,
            centre_x,
            centre_y,
            &mut collider_set,
            half_width,
            half_height,
        );
        joint_between_rigid_bodies(
            wall_handle,
            point![wall_width, 0.0],
            body_mb.rb,
            point![-half_width, 0.0],
            &mut impulse_joint_set,
        );

        let mut physics_pipeline = PhysicsPipeline::new();
        let mut integration_params = IntegrationParameters::default();
        integration_params.dt = 1.0 / 240.0;
        integration_params.max_ccd_substeps = 4;
        let mut island_manager = IslandManager::new();
        let mut broad_phase = DefaultBroadPhase::new();
        let mut narrow_phase = NarrowPhase::new();
        let mut multi_body_joint_set = MultibodyJointSet::new();
        let mut ccd_solver = CCDSolver::new();

        for _ in 0..100 {
            let collider_pos =
                body_mb.get_bounding_box(&rigid_body_set);

            physics_pipeline.step(
                &vector![0.0, -9.81],
                &integration_params,
                &mut island_manager,
                &mut broad_phase,
                &mut narrow_phase,
                &mut rigid_body_set,
                &mut collider_set,
                &mut impulse_joint_set,
                &mut multi_body_joint_set,
                &mut ccd_solver,
                &(),
                &(),
            );

            assert!((distance(&collider_pos[0],&collider_pos[1])- half_width*2.).abs() < 0.001);
            assert!((distance(&collider_pos[1],&collider_pos[2])- half_height*2.).abs() < 0.001);
        }
    }
}
