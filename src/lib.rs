use rapier2d::na::{Point2, Vector2};
use rapier2d::prelude::*;

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
        let wall_rb = rigid_body_set.get(wall_handle).unwrap();
        let wall_middle_y = wall_rb.translation().y;
        let wall_x = wall_rb.translation().x;

        // Calculate positions based on wall position and component dimensions
        let wall_right_edge = wall_x + WALL_HALF_WIDTH;
        let tricep_x = wall_right_edge + TRICEP_TO_WALL_SPACING + TRICEP_HALF_WIDTH;
        let forearm_x = tricep_x + TRICEP_HALF_WIDTH + FOREARM_HALF_WIDTH + TRICEP_TO_FOREARM_SPACING;
        let palm_x = forearm_x + FOREARM_HALF_WIDTH + PALM_HALF_WIDTH + FOREARM_TO_PALM_SPACING;
        let lower_finger_x = palm_x + PALM_HALF_WIDTH + FINGER_HALF_WIDTH + PALM_TO_FINGER_SPACING;
        let upper_finger_x = lower_finger_x + FINGER_HALF_WIDTH + FINGER_HALF_WIDTH + FINGER_SEGMENT_SPACING;

        // Tricep
        let tricep_body = RigidBodyBuilder::dynamic()
            .translation(vector![tricep_x, wall_middle_y])
            .can_sleep(false)
            .ccd_enabled(true)
            .build();
        let tricep_handle = rigid_body_set.insert(tricep_body);

        let tricep_collider = ColliderBuilder::cuboid(TRICEP_HALF_WIDTH, TRICEP_HALF_HEIGHT)
            .restitution(0.7)
            .friction(0.3)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .build();
        collider_set.insert_with_parent(tricep_collider, tricep_handle, rigid_body_set);

        let shoulder_joint = RevoluteJointBuilder::new()
            .local_anchor1(WALL_SHOULDER_ANCHOR)
            .local_anchor2(TRICEP_SHOULDER_ANCHOR)
            .build();
        impulse_joint_set.insert(wall_handle, tricep_handle, shoulder_joint, true);

        // Forearm
        let forearm_body = RigidBodyBuilder::dynamic()
            .translation(vector![forearm_x, wall_middle_y])
            .can_sleep(false)
            .ccd_enabled(true)
            .build();
        let forearm_handle = rigid_body_set.insert(forearm_body);

        let forearm_collider = ColliderBuilder::cuboid(FOREARM_HALF_WIDTH, FOREARM_HALF_HEIGHT)
            .restitution(0.7)
            .friction(0.3)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .build();
        collider_set.insert_with_parent(forearm_collider, forearm_handle, rigid_body_set);

        let elbow_joint = RevoluteJointBuilder::new()
            .local_anchor1(TRICEP_ELBOW_ANCHOR)
            .local_anchor2(FOREARM_ELBOW_ANCHOR)
            .build();
        impulse_joint_set.insert(tricep_handle, forearm_handle, elbow_joint, true);

        // Palm
        let palm_body = RigidBodyBuilder::dynamic()
            .translation(vector![palm_x, wall_middle_y])
            .can_sleep(false)
            .ccd_enabled(true)
            .build();
        let palm_handle = rigid_body_set.insert(palm_body);

        let palm_collider = ColliderBuilder::cuboid(PALM_HALF_WIDTH, PALM_HALF_HEIGHT)
            .restitution(0.7)
            .friction(0.3)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .build();
        collider_set.insert_with_parent(palm_collider, palm_handle, rigid_body_set);

        let wrist_joint = RevoluteJointBuilder::new()
            .local_anchor1(FOREARM_WRIST_ANCHOR)
            .local_anchor2(PALM_WRIST_ANCHOR)
            .build();
        impulse_joint_set.insert(forearm_handle, palm_handle, wrist_joint, true);

        // Lower index finger
        let lower_index_finger_body = RigidBodyBuilder::dynamic()
            .translation(vector![lower_finger_x, wall_middle_y])
            .can_sleep(false)
            .ccd_enabled(true)
            .build();
        let lower_index_finger_handle = rigid_body_set.insert(lower_index_finger_body);

        let lower_index_finger_collider = ColliderBuilder::cuboid(FINGER_HALF_WIDTH, FINGER_HALF_HEIGHT)
            .restitution(0.7)
            .friction(0.3)
            .build();
        collider_set.insert_with_parent(lower_index_finger_collider, lower_index_finger_handle, rigid_body_set);

        // Upper index finger
        let upper_index_finger_body = RigidBodyBuilder::dynamic()
            .translation(vector![upper_finger_x, wall_middle_y])
            .can_sleep(false)
            .ccd_enabled(true)
            .build();
        let upper_index_finger_handle = rigid_body_set.insert(upper_index_finger_body);

        let upper_index_finger_collider = ColliderBuilder::cuboid(FINGER_HALF_WIDTH, FINGER_HALF_HEIGHT)
            .restitution(0.7)
            .friction(0.3)
            .build();
        collider_set.insert_with_parent(upper_index_finger_collider, upper_index_finger_handle, rigid_body_set);

        // Index finger joints
        let palm_index_finger_joint = RevoluteJointBuilder::new()
            .local_anchor1(PALM_INDEX_ANCHOR)
            .local_anchor2(FINGER_JOINT_ANCHOR_LEFT)
            .build();
        impulse_joint_set.insert(palm_handle, lower_index_finger_handle, palm_index_finger_joint, true);

        let middle_index_finger_joint = RevoluteJointBuilder::new()
            .local_anchor1(FINGER_JOINT_ANCHOR)
            .local_anchor2(FINGER_JOINT_ANCHOR_LEFT)
            .build();
        impulse_joint_set.insert(lower_index_finger_handle, upper_index_finger_handle, middle_index_finger_joint, true);

        // Lower thumb
        let lower_thumb_body = RigidBodyBuilder::dynamic()
            .translation(vector![palm_x, wall_middle_y + PALM_TO_THUMB_OFFSET_Y])
            .can_sleep(false)
            .ccd_enabled(true)
            .build();
        let lower_thumb_handle = rigid_body_set.insert(lower_thumb_body);

        let lower_thumb_collider = ColliderBuilder::cuboid(THUMB_HALF_WIDTH, THUMB_HALF_HEIGHT)
            .restitution(0.7)
            .friction(0.3)
            .build();
        collider_set.insert_with_parent(lower_thumb_collider, lower_thumb_handle, rigid_body_set);

        // Upper thumb
        let upper_thumb_body = RigidBodyBuilder::dynamic()
            .translation(vector![palm_x, wall_middle_y + PALM_TO_THUMB_OFFSET_Y + THUMB_SEGMENT_SPACING])
            .can_sleep(false)
            .ccd_enabled(true)
            .build();
        let upper_thumb_handle = rigid_body_set.insert(upper_thumb_body);

        let upper_thumb_collider = ColliderBuilder::cuboid(THUMB_HALF_WIDTH, THUMB_HALF_HEIGHT)
            .restitution(0.7)
            .friction(0.3)
            .build();
        collider_set.insert_with_parent(upper_thumb_collider, upper_thumb_handle, rigid_body_set);

        // Thumb joints
        let palm_thumb_joint = RevoluteJointBuilder::new()
            .local_anchor1(PALM_THUMB_ANCHOR)
            .local_anchor2(THUMB_JOINT_ANCHOR_TOP)
            .build();
        impulse_joint_set.insert(palm_handle, lower_thumb_handle, palm_thumb_joint, true);

        let middle_thumb_joint = RevoluteJointBuilder::new()
            .local_anchor1(THUMB_JOINT_ANCHOR_BOTTOM)
            .local_anchor2(THUMB_JOINT_ANCHOR_TOP)
            .build();
        impulse_joint_set.insert(lower_thumb_handle, upper_thumb_handle, middle_thumb_joint, true);

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
                            name,
                            aabb.mins.x,
                            aabb.mins.y,
                            aabb.maxs.x,
                            aabb.maxs.y
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
        // Get the rigid body
        let rigid_body = rigid_body_set.get(rigid_body_handle)?;

        // Find the collider attached to this rigid body
        let collider = collider_set
            .iter()
            .find(|(_, collider)| collider.parent() == Some(rigid_body_handle))
            .map(|(_, collider)| collider)?;

        // Get the collider's shape as a cuboid
        let cuboid = collider.shape().as_cuboid()?;
        let half_extents = cuboid.half_extents;

        // Get the rigid body's transform
        let rigid_body_pos = rigid_body.position();

        // Calculate the 4 corners of the cuboid in local coordinates
        let local_corners = [
            Point2::new(-half_extents.x, -half_extents.y), // bottom-left
            Point2::new(half_extents.x, -half_extents.y),  // bottom-right
            Point2::new(half_extents.x, half_extents.y),   // top-right
            Point2::new(-half_extents.x, half_extents.y),  // top-left
        ];

        // Transform corners to world coordinates
        let world_corners: Vec<Point2<f32>> = local_corners
            .iter()
            .map(|&local_corner| {
                rigid_body_pos.rotation.transform_point(&local_corner) + rigid_body_pos.translation.vector
            })
            .collect();

        // Calculate the rigid body's center in world coordinates
        let body_center = rigid_body_pos.translation.vector;

        // Find the direction from joint to body center (this is the main axis)
        let joint_to_center = body_center - joint_position.coords;
        let main_axis_direction = if joint_to_center.norm() > f32::EPSILON {
            joint_to_center.normalize()
        } else {
            // Fallback: use the rigid body's local x-axis as main direction
            rigid_body_pos.rotation.transform_vector(&Vector2::new(1.0, 0.0))
        };

        // Project each corner onto the main axis and find the two farthest
        let mut corner_projections: Vec<(Point2<f32>, f32)> = world_corners
            .into_iter()
            .map(|corner| {
                let corner_vec = corner - joint_position;
                let projection = corner_vec.dot(&main_axis_direction);
                (corner, projection)
            })
            .collect();

        // Sort by projection distance along the main axis
        corner_projections.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Get the two corners that are farthest along the main axis
        let farthest_corner = corner_projections.last()?.0;

        // For the second corner, find the one that's also far along the axis but has different y-coordinate
        // This ensures we get the "upper" and "lower" corners of the far end
        let farthest_projection = corner_projections.last()?.1;
        let second_farthest = corner_projections
            .iter()
            .rev()
            .skip(1) // Skip the absolute farthest
            .find(|(corner, projection)| {
                // Look for a corner that's close in projection distance but different in position
                (projection - farthest_projection).abs() < half_extents.norm() * 0.5 &&
                    (corner.y - farthest_corner.y).abs() > f32::EPSILON
            })
            .map(|(corner, _)| *corner)
            .unwrap_or_else(|| {
                // Fallback: just take the second farthest by projection
                corner_projections[corner_projections.len() - 2].0
            });

        // Determine which is upper and which is lower based on y-coordinate
        let (upper_corner, lower_corner) = if farthest_corner.y > second_farthest.y {
            (farthest_corner, second_farthest)
        } else {
            (second_farthest, farthest_corner)
        };

        Some((
            (upper_corner.x, upper_corner.y),
            (lower_corner.x, lower_corner.y),
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
        let wall_joint_pos = if let Some(wall_rb) = rigid_body_set.iter().find(|(_, rb)| rb.body_type() == RigidBodyType::Fixed).map(|(_, rb)| rb) {
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

        Self::farthest_corners_from_joint(self.tricep_handle, wall_joint_pos, rigid_body_set, collider_set)
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
            tricep_pos.rotation.transform_point(&TRICEP_ELBOW_ANCHOR) + tricep_pos.translation.vector
        } else {
            // Fallback: assume elbow is at forearm's left anchor position
            if let Some(forearm_rb) = rigid_body_set.get(self.forearm_handle) {
                Point2::new(forearm_rb.translation().x + FOREARM_ELBOW_ANCHOR.x, forearm_rb.translation().y)
            } else {
                return None;
            }
        };

        Self::farthest_corners_from_joint(self.forearm_handle, elbow_joint_pos, rigid_body_set, collider_set)
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
            forearm_pos.rotation.transform_point(&FOREARM_WRIST_ANCHOR) + forearm_pos.translation.vector
        } else {
            // Fallback: assume wrist is at palm's left anchor position
            if let Some(palm_rb) = rigid_body_set.get(self.palm_handle) {
                Point2::new(palm_rb.translation().x + PALM_WRIST_ANCHOR.x, palm_rb.translation().y)
            } else {
                return None;
            }
        };

        Self::farthest_corners_from_joint(self.palm_handle, wrist_joint_pos, rigid_body_set, collider_set)
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
                Point2::new(finger_rb.translation().x + FINGER_JOINT_ANCHOR_LEFT.x, finger_rb.translation().y)
            } else {
                return None;
            }
        };

        Self::farthest_corners_from_joint(self.lower_index_finger_handle, palm_joint_pos, rigid_body_set, collider_set)
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
        let middle_joint_pos = if let Some(lower_finger_rb) = rigid_body_set.get(self.lower_index_finger_handle) {
            let lower_finger_pos = lower_finger_rb.position();
            lower_finger_pos.rotation.transform_point(&FINGER_JOINT_ANCHOR) + lower_finger_pos.translation.vector
        } else {
            // Fallback: assume joint is at upper finger's left anchor position
            if let Some(upper_finger_rb) = rigid_body_set.get(self.upper_index_finger_handle) {
                Point2::new(upper_finger_rb.translation().x + FINGER_JOINT_ANCHOR_LEFT.x, upper_finger_rb.translation().y)
            } else {
                return None;
            }
        };

        Self::farthest_corners_from_joint(self.upper_index_finger_handle, middle_joint_pos, rigid_body_set, collider_set)
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
                Point2::new(thumb_rb.translation().x, thumb_rb.translation().y + THUMB_JOINT_ANCHOR_TOP.y)
            } else {
                return None;
            }
        };

        Self::farthest_corners_from_joint(self.lower_thumb_handle, palm_joint_pos, rigid_body_set, collider_set)
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
        let middle_joint_pos = if let Some(lower_thumb_rb) = rigid_body_set.get(self.lower_thumb_handle) {
            let lower_thumb_pos = lower_thumb_rb.position();
            lower_thumb_pos.rotation.transform_point(&THUMB_JOINT_ANCHOR_BOTTOM) + lower_thumb_pos.translation.vector
        } else {
            // Fallback: assume joint is at upper thumb's top anchor position
            if let Some(upper_thumb_rb) = rigid_body_set.get(self.upper_thumb_handle) {
                Point2::new(upper_thumb_rb.translation().x, upper_thumb_rb.translation().y + THUMB_JOINT_ANCHOR_TOP.y)
            } else {
                return None;
            }
        };

        Self::farthest_corners_from_joint(self.upper_thumb_handle, middle_joint_pos, rigid_body_set, collider_set)
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
        let target_body_pos = rigid_body_set.get(target_handle).unwrap().center_of_mass();

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
        let target_rb = rigid_body_set.get_mut(target_handle).unwrap();
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
    pub fn apply_palm_force(
        &self,
        scaling_factor: f32,
        rigid_body_set: &mut RigidBodySet,
    ) -> bool {
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
    wall_handle: RigidBodyHandle,
    ground_handle: RigidBodyHandle,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();
        let mut impulse_joint_set = ImpulseJointSet::new();

        // Create the ground
        let ground_y = -2.0;
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
            wall_handle,
            ground_handle,
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
            None,
            &physics_hooks,
            &event_handler,
        );
    }

    /// Prints the current state of all arm components
    pub fn print_arm_state(&self) {
        self.arm.print_state(&self.rigid_body_set, &self.collider_set);
    }

    // Force application methods
    pub fn apply_tricep_force(&mut self, scaling_factor: f32) -> bool {
        self.arm.apply_tricep_force(scaling_factor, &mut self.rigid_body_set)
    }

    pub fn apply_forearm_force(&mut self, scaling_factor: f32) -> bool {
        self.arm.apply_forearm_force(scaling_factor, &mut self.rigid_body_set)
    }

    pub fn apply_palm_force(&mut self, scaling_factor: f32) -> bool {
        self.arm.apply_palm_force(scaling_factor, &mut self.rigid_body_set)
    }

    pub fn apply_lower_index_finger_force(&mut self, scaling_factor: f32) -> bool {
        self.arm.apply_lower_index_finger_force(scaling_factor, &mut self.rigid_body_set)
    }

    pub fn apply_upper_index_finger_force(&mut self, scaling_factor: f32) -> bool {
        self.arm.apply_upper_index_finger_force(scaling_factor, &mut self.rigid_body_set)
    }

    pub fn apply_lower_thumb_force(&mut self, scaling_factor: f32) -> bool {
        self.arm.apply_lower_thumb_force(scaling_factor, &mut self.rigid_body_set)
    }

    pub fn apply_upper_thumb_force(&mut self, scaling_factor: f32) -> bool {
        self.arm.apply_upper_thumb_force(scaling_factor, &mut self.rigid_body_set)
    }

    // Farthest corners query methods
    pub fn tricep_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm.tricep_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

    pub fn forearm_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm.forearm_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

    pub fn palm_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm.palm_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

    pub fn lower_index_finger_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm.lower_index_finger_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

    pub fn upper_index_finger_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm.upper_index_finger_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

    pub fn lower_thumb_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm.lower_thumb_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

    pub fn upper_thumb_farthest_corners(&self) -> Option<((f32, f32), (f32, f32))> {
        self.arm.upper_thumb_farthest_corners(&self.rigid_body_set, &self.collider_set)
    }

}


pub fn create_physics_world() {
    let mut physics_world = PhysicsWorld::new();

    // Run the simulation
    for step in 0..=200 {
        if step % 20 == 0 {
            println!("Step {}", step);
            physics_world.print_arm_state();

            // Print tricep's farthest corners
            if let Some(((upper_x, upper_y), (lower_x, lower_y))) = physics_world.tricep_farthest_corners() {
                println!(
                    "Tricep farthest corners: upper=({:.3}, {:.3}), lower=({:.3}, {:.3})",
                    upper_x, upper_y, lower_x, lower_y
                );
            } else {
                println!("Tricep farthest corners: Could not calculate");
            }

            // Print forearm's farthest corners
            if let Some(((upper_x, upper_y), (lower_x, lower_y))) = physics_world.forearm_farthest_corners() {
                println!(
                    "Forearm farthest corners: upper=({:.3}, {:.3}), lower=({:.3}, {:.3})",
                    upper_x, upper_y, lower_x, lower_y
                );
            } else {
                println!("Forearm farthest corners: Could not calculate");
            }
        }

        // Apply a small force to the tricep to create movement
        if step < 50 {
            physics_world.apply_tricep_force(0.4); // 40% force away from wall
        }

        physics_world.step();
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