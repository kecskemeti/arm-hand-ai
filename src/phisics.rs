use rapier2d::dynamics::{
    CCDSolver, ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet,
    RevoluteJointBuilder, RigidBodyBuilder, RigidBodyHandle, RigidBodySet, RigidBodyType,
};
use rapier2d::geometry::{ColliderBuilder, ColliderSet, DefaultBroadPhase, NarrowPhase};
use rapier2d::na::{point, vector, Point2, Vector2};
use rapier2d::pipeline::{ActiveEvents, PhysicsPipeline};
use rapier2d::prelude::nalgebra;

// Ground dimensions
const GROUND_HALF_WIDTH: f32 = 100.0;
const GROUND_HALF_HEIGHT: f32 = 0.1;

// Wall dimensions
const WALL_HALF_WIDTH: f32 = 0.2;
const WALL_HALF_HEIGHT: f32 = 2.0;

// Cuboid dimensions (half-extents)
const TRICEP_HALF_WIDTH: f32 = 0.25;
const TRICEP_HALF_HEIGHT: f32 = 0.04;

const FOREARM_HALF_WIDTH: f32 = 0.25;
const FOREARM_HALF_HEIGHT: f32 = 0.035;

const PALM_HALF_WIDTH: f32 = 0.05;
const PALM_HALF_HEIGHT: f32 = 0.015;

const FINGER_HALF_WIDTH: f32 = 0.025;
const FINGER_HALF_HEIGHT: f32 = 0.01;

const THUMB_HALF_WIDTH: f32 = 0.01;
const THUMB_HALF_HEIGHT: f32 = 0.025;

// Joint anchor points (relative to cuboid centers)
const WALL_SHOULDER_ANCHOR: Point2<f32> = point![WALL_HALF_WIDTH, 0.0];
const TRICEP_SHOULDER_ANCHOR: Point2<f32> = point![-TRICEP_HALF_WIDTH * 2.0, 0.0]; // At left edge
const TRICEP_ELBOW_ANCHOR: Point2<f32> = point![TRICEP_HALF_WIDTH * 2.0, 0.0]; // At right edge
const FOREARM_ELBOW_ANCHOR: Point2<f32> = point![-FOREARM_HALF_WIDTH * 1.6, 0.0]; // Slightly inward from left edge
const FOREARM_WRIST_ANCHOR: Point2<f32> = point![FOREARM_HALF_WIDTH * 1.6, 0.0]; // Slightly inward from right edge
const PALM_WRIST_ANCHOR: Point2<f32> = point![-PALM_HALF_WIDTH * 5.0, 0.0]; // At left edge
const PALM_INDEX_ANCHOR: Point2<f32> = point![PALM_HALF_WIDTH, 0.0]; // At right edge
const PALM_THUMB_ANCHOR: Point2<f32> = point![0.0, -PALM_HALF_HEIGHT]; // At bottom edge

const FINGER_JOINT_ANCHOR: Point2<f32> = point![FINGER_HALF_WIDTH, 0.0]; // At right edge
const FINGER_JOINT_ANCHOR_LEFT: Point2<f32> = point![-FINGER_HALF_WIDTH, 0.0]; // At left edge

const THUMB_JOINT_ANCHOR_BOTTOM: Point2<f32> = point![0.0, -THUMB_HALF_HEIGHT * 0.4]; // Near bottom
const THUMB_JOINT_ANCHOR_TOP: Point2<f32> = point![0.0, THUMB_HALF_HEIGHT * 0.4]; // Near top

// Spacing between connected rigid bodies
const TRICEP_TO_WALL_SPACING: f32 = 0.02; // Gap between wall and tricep
const TRICEP_TO_FOREARM_SPACING: f32 = 0.015; // Small gap between tricep and forearm
const FOREARM_TO_PALM_SPACING: f32 = 0.015; // Small gap between forearm and palm
const PALM_TO_FINGER_SPACING: f32 = 0.005; // Small gap between palm and finger segments
const FINGER_SEGMENT_SPACING: f32 = 0.005; // Small gap between finger segments
const PALM_TO_THUMB_OFFSET_Y: f32 = -0.05; // Thumb offset below palm
const THUMB_SEGMENT_SPACING: f32 = -0.06; // Vertical spacing between thumb segments

const MAX_X: f32 = WALL_HALF_WIDTH
    + TRICEP_TO_WALL_SPACING
    + TRICEP_HALF_WIDTH * 2.0
    + TRICEP_TO_FOREARM_SPACING
    + FOREARM_HALF_WIDTH * 2.0
    + FOREARM_TO_PALM_SPACING
    + PALM_HALF_WIDTH * 2.0
    + PALM_TO_FINGER_SPACING
    + FINGER_HALF_WIDTH * 2.0
    + FINGER_SEGMENT_SPACING
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

fn create_body_and_cub_collider(
    body_set: &mut RigidBodySet,
    centre_x: f32,
    centre_y: f32,
    collider_set: &mut ColliderSet,
    width: f32,
    height: f32,
) -> RigidBodyHandle {
    let body_handle = create_dynamic_body(body_set, centre_x, centre_y);
    let collider_handle = ColliderBuilder::cuboid(width, height)
        .restitution(0.7)
        .friction(0.3)
        .active_events(ActiveEvents::COLLISION_EVENTS)
        .build();
    collider_set.insert_with_parent(collider_handle, body_handle, body_set);
    body_handle
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

fn join_horizontal_rigid_bodies(
    rb1: RigidBodyHandle,
    rb2: RigidBodyHandle,
    rigid_body_set: &RigidBodySet,
    collider_set: &ColliderSet,
    joint_set: &mut ImpulseJointSet,
) {
    // Get rigid bodies
    let body1 = &rigid_body_set[rb1];
    let body2 = &rigid_body_set[rb2];

    // Get colliders (assuming one per body)
    let collider1 = collider_set.get(body1.colliders()[0]).unwrap();
    let collider2 = collider_set.get(body2.colliders()[0]).unwrap();

    // Get cuboid half extents
    let half_extents1 = match collider1.shape().as_cuboid() {
        Some(cuboid) => cuboid.half_extents,
        None => panic!("Collider1 is not a cuboid"),
    };

    let half_extents2 = match collider2.shape().as_cuboid() {
        Some(cuboid) => cuboid.half_extents,
        None => panic!("Collider2 is not a cuboid"),
    };

    // Compute local anchor points
    let local_anchor1 = point![half_extents1.x, 0.0]; // Right middle of collider1
    let local_anchor2 = point![-half_extents2.x, 0.0]; // Left middle of collider2

    // Build and insert revolute joint
    let joint = RevoluteJointBuilder::new()
        .local_anchor1(local_anchor1)
        .local_anchor2(local_anchor2)
        .build();

    joint_set.insert(rb1, rb2, joint, true);
}

fn get_cuboid_collider_corners(
    rb: RigidBodyHandle,
    rigid_body_set: &RigidBodySet,
    collider_set: &ColliderSet,
) -> [Point2<f32>; 4] {
    let body = &rigid_body_set.get(rb).unwrap();
    let collider = collider_set.get(body.colliders()[0]).unwrap();
    let cuboid = collider.shape().as_cuboid().unwrap();
    let half_extents = cuboid.half_extents;
    let local_corners = [
        point![-half_extents.x, -half_extents.y],
        point![half_extents.x, -half_extents.y],
        point![half_extents.x, half_extents.y],
        point![-half_extents.x, half_extents.y],
    ];
    let rb_pos = body.position();
    let world_transform = *rb_pos;

    local_corners
        .iter()
        .map(|p| world_transform * p)
        .collect::<Vec<Point2<f32>>>()
        .try_into()
        .unwrap()
}

pub struct Arm {
    tricep_handle: RigidBodyHandle,
    forearm_handle: RigidBodyHandle,
    palm_handle: RigidBodyHandle,
    lower_index_finger_handle: RigidBodyHandle,
    upper_index_finger_handle: RigidBodyHandle,
    lower_thumb_handle: RigidBodyHandle,
    upper_thumb_handle: RigidBodyHandle,
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
        let tricep_x = wall_right_edge + TRICEP_TO_WALL_SPACING + TRICEP_HALF_WIDTH;
        let forearm_x =
            tricep_x + TRICEP_HALF_WIDTH + FOREARM_HALF_WIDTH + TRICEP_TO_FOREARM_SPACING;
        let palm_x = forearm_x + FOREARM_HALF_WIDTH + PALM_HALF_WIDTH + FOREARM_TO_PALM_SPACING;
        let lower_finger_x = palm_x + PALM_HALF_WIDTH + FINGER_HALF_WIDTH + PALM_TO_FINGER_SPACING;
        let upper_finger_x =
            lower_finger_x + FINGER_HALF_WIDTH + FINGER_HALF_WIDTH + FINGER_SEGMENT_SPACING;
        println!("{wall_middle_y}");
        println!("{wall_x}");
        println!("{wall_right_edge}");
        println!("{tricep_x}");
        println!("{forearm_x}");
        println!("{palm_x}");
        println!("{lower_finger_x}");
        println!("{upper_finger_x}");

        // Tricep
        let tricep_handle = create_body_and_cub_collider(
            rigid_body_set,
            tricep_x,
            wall_middle_y,
            collider_set,
            TRICEP_HALF_WIDTH,
            TRICEP_HALF_HEIGHT,
        );
        println!(
            "{:?}",
            get_cuboid_collider_corners(tricep_handle, rigid_body_set, collider_set)
        );
        joint_between_rigid_bodies(
            wall_handle,
            WALL_SHOULDER_ANCHOR,
            tricep_handle,
            TRICEP_SHOULDER_ANCHOR,
            impulse_joint_set,
        );

        // Forearm
        let forearm_handle = create_body_and_cub_collider(
            rigid_body_set,
            forearm_x,
            wall_middle_y,
            collider_set,
            FOREARM_HALF_WIDTH,
            FOREARM_HALF_HEIGHT,
        );

        //Elbow:
        join_horizontal_rigid_bodies(
            tricep_handle,
            forearm_handle,
            rigid_body_set,
            collider_set,
            impulse_joint_set,
        );

        // Palm
        let palm_handle = create_body_and_cub_collider(
            rigid_body_set,
            palm_x,
            wall_middle_y,
            collider_set,
            PALM_HALF_WIDTH,
            PALM_HALF_HEIGHT,
        );

        //Wrist:
        join_horizontal_rigid_bodies(
            forearm_handle,
            palm_handle,
            rigid_body_set,
            collider_set,
            impulse_joint_set,
        );

        // Lower index finger
        let lower_index_finger_handle = create_body_and_cub_collider(
            rigid_body_set,
            lower_finger_x,
            wall_middle_y,
            collider_set,
            FINGER_HALF_WIDTH,
            FINGER_HALF_HEIGHT,
        );

        // Index finger lower joint
        join_horizontal_rigid_bodies(
            palm_handle,
            lower_index_finger_handle,
            rigid_body_set,
            collider_set,
            impulse_joint_set,
        );

        // Upper index finger
        let upper_index_finger_handle = create_body_and_cub_collider(
            rigid_body_set,
            upper_finger_x,
            wall_middle_y,
            collider_set,
            FINGER_HALF_WIDTH,
            FINGER_HALF_HEIGHT,
        );

        // Index finger upper joint
        join_horizontal_rigid_bodies(
            lower_index_finger_handle,
            upper_index_finger_handle,
            rigid_body_set,
            collider_set,
            impulse_joint_set,
        );

        // Lower thumb
        let lower_thumb_handle = create_body_and_cub_collider(
            rigid_body_set,
            palm_x,
            wall_middle_y + PALM_TO_THUMB_OFFSET_Y,
            collider_set,
            THUMB_HALF_WIDTH,
            THUMB_HALF_HEIGHT,
        );

        // lower thumb joint
        joint_between_rigid_bodies(
            palm_handle,
            PALM_THUMB_ANCHOR,
            lower_thumb_handle,
            THUMB_JOINT_ANCHOR_TOP,
            impulse_joint_set,
        );

        // Upper thumb
        let upper_thumb_handle = create_body_and_cub_collider(
            rigid_body_set,
            palm_x,
            wall_middle_y + PALM_TO_THUMB_OFFSET_Y + THUMB_SEGMENT_SPACING,
            collider_set,
            THUMB_HALF_WIDTH,
            THUMB_HALF_HEIGHT,
        );

        joint_between_rigid_bodies(
            lower_thumb_handle,
            THUMB_JOINT_ANCHOR_BOTTOM,
            upper_thumb_handle,
            THUMB_JOINT_ANCHOR_TOP,
            impulse_joint_set,
        );

        Self {
            tricep_handle,
            forearm_handle,
            palm_handle,
            lower_index_finger_handle,
            upper_index_finger_handle,
            lower_thumb_handle,
            upper_thumb_handle,
        }
    }

    pub fn all_corners(
        &self,
        rigid_body_set: &RigidBodySet,
        collider_set: &ColliderSet,
    ) -> Vec<[Point2<f32>; 4]> {
        [
            self.tricep_handle,
            self.forearm_handle,
            self.palm_handle,
            self.lower_index_finger_handle,
            self.upper_index_finger_handle,
            self.lower_thumb_handle,
            self.upper_thumb_handle,
        ]
        .iter()
        .map(|&rb_handle| get_cuboid_collider_corners(rb_handle, rigid_body_set, collider_set))
        .collect()
    }

    pub fn print_state(&self, rigid_body_set: &RigidBodySet, collider_set: &ColliderSet) {
        let handles = [
            ("Tricep", self.tricep_handle),
            ("Forearm", self.forearm_handle),
            ("Palm", self.palm_handle),
            ("Lower Index Finger", self.lower_index_finger_handle),
            ("Upper Index Finger", self.upper_index_finger_handle),
            ("Lower Thumb", self.lower_thumb_handle),
            ("Upper Thumb", self.upper_thumb_handle),
        ];
        for (name, handle) in handles {
            if let Some(rb) = rigid_body_set.get(handle) {
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

    /// Gets the upper and lower corners of a rigid body that are furthest from a specified joint position.
    ///
    /// This method considers the rigid body's actual orientation and position, transforming the
    /// local corners into world coordinates to determine the farthest points from the joint.
    ///
    /// Returns a tuple of ((x_up, y_up), (x_low, y_low)) for the world coordinates.
    fn farthest_corners_from_joint(
        rigid_body_handle: RigidBodyHandle,
        joint_position: Point2<f32>,
        rigid_body_set: &RigidBodySet,
        collider_set: &ColliderSet,
    ) -> Option<((f32, f32), (f32, f32))> {
        // Get the rigid body and its collider
        let rigid_body = rigid_body_set.get(rigid_body_handle)?;
        let collider = collider_set
            .iter()
            .find(|(_, collider)| collider.parent() == Some(rigid_body_handle))
            .map(|(_, collider)| collider)?;

        // Get the cuboid and its half-extents
        let cuboid = collider.shape().as_cuboid()?;
        let half_extents = cuboid.half_extents;

        // Get the rigid body's transform
        let rigid_body_pos = rigid_body.position();

        // Transform the joint position into the rigid body's local coordinate space
        let local_joint_pos = rigid_body_pos.inverse() * joint_position;

        // Determine the "opposite" side of the body in local space
        // We do this by checking which side the joint is on (x-axis)
        let local_x_sign = local_joint_pos.x.signum();

        // The two corners on the opposite side will have an x-coordinate with the opposite sign
        let opposite_x = -local_x_sign * half_extents.x;

        // The y-coordinates are the half-extents for the "upper" and "lower" corners
        let local_upper_corner = Point2::new(opposite_x, half_extents.y);
        let local_lower_corner = Point2::new(opposite_x, -half_extents.y);

        // Transform the local corners back to world coordinates
        let world_upper_corner = rigid_body_pos * local_upper_corner;
        let world_lower_corner = rigid_body_pos * local_lower_corner;

        // Return the corners
        Some((
            (world_upper_corner.x, world_upper_corner.y),
            (world_lower_corner.x, world_lower_corner.y),
        ))
    }

    /// Gets the upper and lower corners of the tricep that are furthest from the wall joint.
    ///
    /// This method considers the tricep's actual orientation and position, transforming the
    /// local corners into world coordinates to determine the farthest points.
    ///
    /// Returns a tuple of ((x_up, y_up), (x_low, y_low)) for the world coordinates.
    pub fn tricep_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
        collider_set: &ColliderSet,
    ) -> Option<((f32, f32), (f32, f32))> {
        // Get the wall joint position (shoulder joint anchor point on the wall)
        // The shoulder joint connects at WALL_SHOULDER_ANCHOR on the wall
        let wall_joint_pos = if let Some(wall_rb) = rigid_body_set
            .iter()
            .find(|(_, rb)| rb.body_type() == RigidBodyType::Fixed)
            .map(|(_, rb)| rb)
        {
            let wall_pos = wall_rb.position();
            wall_pos.rotation.transform_point(&WALL_SHOULDER_ANCHOR) + wall_pos.translation.vector
        } else {
            // Fallback: assume wall is at origin with joint at WALL_SHOULDER_ANCHOR
            if let Some(tricep_rb) = rigid_body_set.get(self.tricep_handle) {
                Point2::new(WALL_SHOULDER_ANCHOR.x, tricep_rb.translation().y)
            } else {
                return None;
            }
        };

        Self::farthest_corners_from_joint(
            self.tricep_handle,
            wall_joint_pos,
            rigid_body_set,
            collider_set,
        )
    }

    /// Gets the upper and lower corners of the forearm that are furthest from the elbow joint.
    ///
    /// This method considers the forearm's actual orientation and position, transforming the
    /// local corners into world coordinates to determine the farthest points from the elbow.
    ///
    /// Returns a tuple of ((x_up, y_up), (x_low, y_low)) for the world coordinates.
    pub fn forearm_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
        collider_set: &ColliderSet,
    ) -> Option<((f32, f32), (f32, f32))> {
        // Get the elbow joint position (elbow joint anchor point on the tricep)
        // The elbow joint connects at TRICEP_ELBOW_ANCHOR on the tricep
        let elbow_joint_pos = if let Some(tricep_rb) = rigid_body_set.get(self.tricep_handle) {
            let tricep_pos = tricep_rb.position();
            tricep_pos.rotation.transform_point(&TRICEP_ELBOW_ANCHOR)
                + tricep_pos.translation.vector
        } else {
            // Fallback: assume elbow is at forearm's left anchor position
            if let Some(forearm_rb) = rigid_body_set.get(self.forearm_handle) {
                Point2::new(
                    forearm_rb.translation().x + FOREARM_ELBOW_ANCHOR.x,
                    forearm_rb.translation().y,
                )
            } else {
                return None;
            }
        };

        Self::farthest_corners_from_joint(
            self.forearm_handle,
            elbow_joint_pos,
            rigid_body_set,
            collider_set,
        )
    }

    /// Gets the upper and lower corners of the palm that are furthest from the wrist joint.
    ///
    /// This method considers the palm's actual orientation and position, transforming the
    /// local corners into world coordinates to determine the farthest points from the wrist.
    ///
    /// Returns a tuple of ((x_up, y_up), (x_low, y_low)) for the world coordinates.
    pub fn palm_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
        collider_set: &ColliderSet,
    ) -> Option<((f32, f32), (f32, f32))> {
        // Get the wrist joint position (wrist joint anchor point on the forearm)
        // The wrist joint connects at FOREARM_WRIST_ANCHOR on the forearm
        let wrist_joint_pos = if let Some(forearm_rb) = rigid_body_set.get(self.forearm_handle) {
            let forearm_pos = forearm_rb.position();
            forearm_pos.rotation.transform_point(&FOREARM_WRIST_ANCHOR)
                + forearm_pos.translation.vector
        } else {
            // Fallback: assume wrist is at palm's left anchor position
            if let Some(palm_rb) = rigid_body_set.get(self.palm_handle) {
                Point2::new(
                    palm_rb.translation().x + PALM_WRIST_ANCHOR.x,
                    palm_rb.translation().y,
                )
            } else {
                return None;
            }
        };

        Self::farthest_corners_from_joint(
            self.palm_handle,
            wrist_joint_pos,
            rigid_body_set,
            collider_set,
        )
    }

    /// Gets the upper and lower corners of the lower index finger that are furthest from the palm joint.
    ///
    /// This method considers the lower index finger's actual orientation and position, transforming the
    /// local corners into world coordinates to determine the farthest points from the palm joint.
    ///
    /// Returns a tuple of ((x_up, y_up), (x_low, y_low)) for the world coordinates.
    pub fn lower_index_finger_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
        collider_set: &ColliderSet,
    ) -> Option<((f32, f32), (f32, f32))> {
        // Get the palm-index finger joint position (palm-index finger joint anchor point on the palm)
        // The joint connects at PALM_INDEX_ANCHOR on the palm
        let palm_joint_pos = if let Some(palm_rb) = rigid_body_set.get(self.palm_handle) {
            let palm_pos = palm_rb.position();
            palm_pos.rotation.transform_point(&PALM_INDEX_ANCHOR) + palm_pos.translation.vector
        } else {
            // Fallback: assume joint is at finger's left anchor position
            if let Some(finger_rb) = rigid_body_set.get(self.lower_index_finger_handle) {
                Point2::new(
                    finger_rb.translation().x + FINGER_JOINT_ANCHOR_LEFT.x,
                    finger_rb.translation().y,
                )
            } else {
                return None;
            }
        };

        Self::farthest_corners_from_joint(
            self.lower_index_finger_handle,
            palm_joint_pos,
            rigid_body_set,
            collider_set,
        )
    }

    /// Gets the upper and lower corners of the upper index finger that are furthest from the middle joint.
    ///
    /// This method considers the upper index finger's actual orientation and position, transforming the
    /// local corners into world coordinates to determine the farthest points from the middle joint.
    ///
    /// Returns a tuple of ((x_up, y_up), (x_low, y_low)) for the world coordinates.
    pub fn upper_index_finger_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
        collider_set: &ColliderSet,
    ) -> Option<((f32, f32), (f32, f32))> {
        // Get the middle index finger joint position (middle joint anchor point on the lower finger)
        // The joint connects at FINGER_JOINT_ANCHOR on the lower index finger
        let middle_joint_pos =
            if let Some(lower_finger_rb) = rigid_body_set.get(self.lower_index_finger_handle) {
                let lower_finger_pos = lower_finger_rb.position();
                lower_finger_pos
                    .rotation
                    .transform_point(&FINGER_JOINT_ANCHOR)
                    + lower_finger_pos.translation.vector
            } else {
                // Fallback: assume joint is at upper finger's left anchor position
                if let Some(upper_finger_rb) = rigid_body_set.get(self.upper_index_finger_handle) {
                    Point2::new(
                        upper_finger_rb.translation().x + FINGER_JOINT_ANCHOR_LEFT.x,
                        upper_finger_rb.translation().y,
                    )
                } else {
                    return None;
                }
            };

        Self::farthest_corners_from_joint(
            self.upper_index_finger_handle,
            middle_joint_pos,
            rigid_body_set,
            collider_set,
        )
    }

    /// Gets the upper and lower corners of the lower thumb that are furthest from the palm joint.
    ///
    /// This method considers the lower thumb's actual orientation and position, transforming the
    /// local corners into world coordinates to determine the farthest points from the palm joint.
    ///
    /// Returns a tuple of ((x_up, y_up), (x_low, y_low)) for the world coordinates.
    pub fn lower_thumb_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
        collider_set: &ColliderSet,
    ) -> Option<((f32, f32), (f32, f32))> {
        // Get the palm-thumb joint position (palm-thumb joint anchor point on the palm)
        // The joint connects at PALM_THUMB_ANCHOR on the palm
        let palm_joint_pos = if let Some(palm_rb) = rigid_body_set.get(self.palm_handle) {
            let palm_pos = palm_rb.position();
            palm_pos.rotation.transform_point(&PALM_THUMB_ANCHOR) + palm_pos.translation.vector
        } else {
            // Fallback: assume joint is at thumb's top anchor position
            if let Some(thumb_rb) = rigid_body_set.get(self.lower_thumb_handle) {
                Point2::new(
                    thumb_rb.translation().x,
                    thumb_rb.translation().y + THUMB_JOINT_ANCHOR_TOP.y,
                )
            } else {
                return None;
            }
        };

        Self::farthest_corners_from_joint(
            self.lower_thumb_handle,
            palm_joint_pos,
            rigid_body_set,
            collider_set,
        )
    }

    /// Gets the upper and lower corners of the upper thumb that are furthest from the middle joint.
    ///
    /// This method considers the upper thumb's actual orientation and position, transforming the
    /// local corners into world coordinates to determine the farthest points from the middle joint.
    ///
    /// Returns a tuple of ((x_up, y_up), (x_low, y_low)) for the world coordinates.
    pub fn upper_thumb_farthest_corners(
        &self,
        rigid_body_set: &RigidBodySet,
        collider_set: &ColliderSet,
    ) -> Option<((f32, f32), (f32, f32))> {
        // Get the middle thumb joint position (middle joint anchor point on the lower thumb)
        // The joint connects at THUMB_JOINT_ANCHOR_BOTTOM on the lower thumb
        let middle_joint_pos =
            if let Some(lower_thumb_rb) = rigid_body_set.get(self.lower_thumb_handle) {
                let lower_thumb_pos = lower_thumb_rb.position();
                lower_thumb_pos
                    .rotation
                    .transform_point(&THUMB_JOINT_ANCHOR_BOTTOM)
                    + lower_thumb_pos.translation.vector
            } else {
                // Fallback: assume joint is at upper thumb's top anchor position
                if let Some(upper_thumb_rb) = rigid_body_set.get(self.upper_thumb_handle) {
                    Point2::new(
                        upper_thumb_rb.translation().x,
                        upper_thumb_rb.translation().y + THUMB_JOINT_ANCHOR_TOP.y,
                    )
                } else {
                    return None;
                }
            };

        Self::farthest_corners_from_joint(
            self.upper_thumb_handle,
            middle_joint_pos,
            rigid_body_set,
            collider_set,
        )
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
            self.tricep_handle,
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
            self.forearm_handle,
            self.tricep_handle,
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
            self.palm_handle,
            self.forearm_handle,
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
            self.lower_index_finger_handle,
            self.palm_handle,
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
            self.upper_index_finger_handle,
            self.lower_index_finger_handle,
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
            self.lower_thumb_handle,
            self.palm_handle,
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
            self.upper_thumb_handle,
            self.lower_thumb_handle,
            THUMB_JOINT_ANCHOR_BOTTOM,
            vector![0.0, 0.05],
            1.0, // Smallest force for thumb tip
            scaling_factor,
            rigid_body_set,
        )
    }

    /// Gets the handle for the upper thumb segment
    pub fn upper_thumb_handle(&self) -> RigidBodyHandle {
        self.upper_thumb_handle
    }

    /// Gets the handle for the upper index finger segment
    pub fn upper_index_finger_handle(&self) -> RigidBodyHandle {
        self.upper_index_finger_handle
    }

    /// Gets handles for all arm segments
    pub fn all_handles(&self) -> [RigidBodyHandle; 7] {
        [
            self.tricep_handle,
            self.forearm_handle,
            self.palm_handle,
            self.lower_index_finger_handle,
            self.upper_index_finger_handle,
            self.lower_thumb_handle,
            self.upper_thumb_handle,
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
    pub fn tricep_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm
            .tricep_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

    pub fn forearm_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm
            .forearm_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

    pub fn palm_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm
            .palm_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

    pub fn lower_index_finger_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm
            .lower_index_finger_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

    pub fn upper_index_finger_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm
            .upper_index_finger_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

    pub fn lower_thumb_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm
            .lower_thumb_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

    pub fn upper_thumb_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm
            .upper_thumb_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

    pub fn all_arm_corners(&self) -> Vec<[Point2<f32>; 4]> {
        self.arm
            .all_corners(&self.rigid_body_set, &self.collider_set)
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
    use crate::phisics::{
        create_body_and_cub_collider, get_cuboid_collider_corners, joint_between_rigid_bodies,
        PhysicsWorld,
    };
    use rapier2d::dynamics::{
        CCDSolver, IntegrationParameters, IslandManager, RigidBodyBuilder, RigidBodySet,
    };
    use rapier2d::geometry::{ColliderBuilder, ColliderSet};
    use rapier2d::na::{point, vector};
    use rapier2d::prelude::{nalgebra, MultibodyJointSet, NarrowPhase, PhysicsPipeline};
    use rapier2d::prelude::{DefaultBroadPhase, ImpulseJointSet};

    #[test]
    fn test_physics_simulation() {
        let mut world = PhysicsWorld::new();
        for _ in 0..10 {
            println!("{:?}", world.all_arm_corners());
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
        let body_handle = create_body_and_cub_collider(
            &mut rigid_body_set,
            centre_x,
            centre_y,
            &mut collider_set,
            half_width,
            half_height,
        );
        let body_pos = rigid_body_set[body_handle].position().translation;
        assert_eq!(body_pos.x, centre_x);
        assert_eq!(body_pos.y, centre_y);
        let collider_pos = get_cuboid_collider_corners(body_handle, &rigid_body_set, &collider_set);
        assert_eq!(collider_pos[0].x, centre_x - half_width);
        assert_eq!(collider_pos[0].y, centre_y - half_height);
        assert_eq!(collider_pos[1].x, centre_x + half_width);
        assert_eq!(collider_pos[1].y, centre_y - half_height);
        assert_eq!(collider_pos[2].x, centre_x + half_width);
        assert_eq!(collider_pos[2].y, centre_y + half_height);
        assert_eq!(collider_pos[3].x, centre_x - half_width);
        assert_eq!(collider_pos[3].y, centre_y + half_height);
    }

    #[test]
    fn bodyies_with_joint_define() {
        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();
        let mut impulse_joint_set = ImpulseJointSet::new();
        let centre_x = 0.47;
        let centre_y = 0.1;
        let half_width = 0.25;
        let half_height = 0.04;
        let wall_rigid_body = RigidBodyBuilder::fixed()
            .translation(vector![0.0, 0.1])
            .build();
        let wall_handle = rigid_body_set.insert(wall_rigid_body);

        let wall_collider = ColliderBuilder::cuboid(0.2, 2.0)
            .restitution(0.7)
            .friction(0.3)
            .build();
        collider_set.insert_with_parent(wall_collider, wall_handle, &mut rigid_body_set);

        let body_handle = create_body_and_cub_collider(
            &mut rigid_body_set,
            centre_x,
            centre_y,
            &mut collider_set,
            half_width,
            half_height,
        );
        joint_between_rigid_bodies(
            wall_handle,
            point![0.2, 0.0],
            body_handle,
            point![-half_width * 2.0, 0.0],
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

        for _ in 0..10 {
            let collider_pos =
                get_cuboid_collider_corners(body_handle, &rigid_body_set, &collider_set);

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

            println!("{:?}", collider_pos);
        }
    }
}
