use std::ops::{Deref, Index};
use rapier2d::dynamics::{ImpulseJointSet, MultibodyJointSet, RevoluteJointBuilder, RigidBodyBuilder, RigidBodyHandle, RigidBodySet};
use rapier2d::geometry::{ColliderBuilder, ColliderSet};
use rapier2d::na::{distance, point, vector, Isometry2, Point2, Vector2};
use rapier2d::prelude::ActiveEvents;
use rapier2d::prelude::nalgebra;
use crate::physics::Corners;
use crate::physics::modelbody::JoinType::*;

#[derive(Default)]
pub(super) struct WorldSets {
    pub(super) rigid_body_set: RigidBodySet,
    pub(super) collider_set: ColliderSet,
    pub(super) impulse_joint_set: ImpulseJointSet,
    pub(super) multibody_joint_set: MultibodyJointSet,
}

#[derive(Debug)]
pub(super) struct BodyStateSnapshot {
    rb: RigidBodyHandle,
    position: Isometry2<f32>,
    linear_velocity: Vector2<f32>,
    angular_velocity: f32,
}

impl BodyStateSnapshot {
    pub(super) fn load(self, body_set: &mut RigidBodySet) {
        let body = &mut body_set[self.rb];
        body.set_position(self.position, true);
        body.set_linvel(self.linear_velocity, true);
        body.set_angvel(self.angular_velocity, true);
    }
}

impl WorldSets {
    pub(super) fn create_joined_body_and_collider(&mut self,
                                       root: &ModelBody,
                                       join: JoinType,
                                       width: f32,
                                       height: f32,
                                       max_force_scale: f32,
    ) -> ModelBody {
        root.create_joined_body_and_collider(
            join,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            width,
            height,
            &mut self.impulse_joint_set,
            max_force_scale
        )
    }

    pub(super) fn create_dynamic_with_cb(&mut self,
                                         centre_x: f32,
                                         centre_y: f32,
                                         width: f32,
                                         height: f32,
                                         cb: ColliderBuilder,
                                         max_force_scale: f32,
    ) -> ModelBody {
        ModelBody::create_dynamic_and_collider(
            &mut self.rigid_body_set, centre_x, centre_y,
            &mut self.collider_set, width, height, cb,
            max_force_scale
        )
    }

    pub(super) fn create_body_with_builders(&mut self,
                                 centre_x: f32,
                                 centre_y: f32,
                                 rbb: RigidBodyBuilder,
                                 width: f32,
                                 height: f32,
                                 cb: ColliderBuilder,
                                 max_force_scale: f32,
    ) -> ModelBody {
        ModelBody::create_body_with_builders(
            &mut self.rigid_body_set,
            centre_x,
            centre_y,
            rbb,
            &mut self.collider_set,
            width,
            height,
            cb,
            max_force_scale
        )
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(super) enum JoinType {
    HorizontalJoin,
    VerticalJoin,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct SingleForcePoint {
    on_body: Point2<f32>,
    around_joint: Point2<f32>
}

impl SingleForcePoint {
    pub fn scaled_force_vector(&self, force:AdjustedForce) -> Vector2<f32> {
        (self.around_joint.coords - self.on_body.coords).normalize() * force.0.abs()
    }

    pub fn transform(&self, tr:&Isometry2<f32>) -> Self {
        Self {
            on_body: self.tr_on_body(tr),
            around_joint: tr * &self.around_joint
        }
    }

    pub fn tr_on_body(&self, tr:&Isometry2<f32>) -> Point2<f32> {
        tr * &self.on_body
    }
}


#[derive(Copy, Clone, Debug, PartialEq)]
struct ForcePoints {
    // backward = towards the previous body
    // forward = towards the next body
    top_backward: SingleForcePoint,
    top_forward: SingleForcePoint,
    bottom_backward: SingleForcePoint,
    bottom_forward: SingleForcePoint,
}

#[derive(Copy, Clone, Debug)]
struct ForceScale {
    scale: f32,
    sigma: f32,
    peak: f32,
}

impl ForceScale {
    pub fn between(forward:&ModelBody, backward:&ModelBody, scale:f32, rigid_body_set: &RigidBodySet) -> AdjustedForce {
        let min_max = forward.max_force_scale.min(backward.max_force_scale);
        let scale = scale.clamp(-1.0, 1.0) * min_max;
        let centre_distances = distance(&forward.starting_centre, &backward.starting_centre);
        let peak = centre_distances/2.;
        let sigma = peak/3.;
        let fs = Self {
            scale,
            sigma,
            peak,
        };
        let fw_tr =rigid_body_set[forward.rb].position();
        let bw_tr =rigid_body_set[backward.rb].position();
        let (fw_anchor, bw_anchor) = if scale > 0. {
            (if let Some(HorizontalJoin) = backward.join_type { forward.force_points.top_forward.tr_on_body(fw_tr) } else {
                forward.force_points.bottom_forward.tr_on_body(fw_tr)
            }, backward.force_points.top_backward.tr_on_body(bw_tr))
        } else {
            (if let Some(HorizontalJoin) = backward.join_type { forward.force_points.bottom_forward.tr_on_body(fw_tr) } else {
                forward.force_points.bottom_backward.tr_on_body(fw_tr)
            }, backward.force_points.bottom_backward.tr_on_body(bw_tr))
        };
        // print!(",{:?},",[point![fw_anchor.x-0.005, fw_anchor.y+0.005], point![fw_anchor.x+0.005, fw_anchor.y+0.005], point![fw_anchor.x+0.005, fw_anchor.y-0.005],point![fw_anchor.x-0.005, fw_anchor.y-0.005]]);
        // println!("{:?}]",[point![bw_anchor.x-0.005, bw_anchor.y+0.005], point![bw_anchor.x+0.005, bw_anchor.y+0.005], point![bw_anchor.x+0.005, bw_anchor.y-0.005],point![bw_anchor.x-0.005, bw_anchor.y-0.005]]);
        fs.adjust(fw_anchor, bw_anchor)
    }

    pub fn adjust(&self, fw_anchor:Point2<f32>, bw_anchor:Point2<f32>) -> AdjustedForce {
        let dist = distance(&fw_anchor, &bw_anchor);
        let exp_base = (dist - self.peak)/self.sigma;
        let exp = exp_base*exp_base/-2.;
        let scaling =exp.exp();
        let adjusted_force = self.scale * scaling;
        AdjustedForce(adjusted_force)
    }
}

#[derive(Copy, Clone, Debug)]
struct AdjustedForce(f32);

impl AdjustedForce {
    pub fn is_upper(&self) -> bool {
        self.0 > 0.0
    }
}

#[derive(Copy, Clone, Debug)]
struct BoundingBox([Point2<f32>; 4]);

impl Deref for BoundingBox {
    type Target = [Point2<f32>; 4];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<[Point2<f32>; 4]> for BoundingBox {
    fn from(points: [Point2<f32>; 4]) -> Self {
        Self(points)
    }
}

impl Index<i8> for BoundingBox {
    type Output = Point2<f32>;
    fn index(&self, index: i8) -> &Self::Output {
        &self.0[((index+4)%4) as usize]
    }
}

fn fractional_point_on_line(p1: Point2<f32>, p2: Point2<f32>, frac:f32) -> Point2<f32> {
    (p1.coords + (p2.coords - p1.coords)*frac).into()
}

fn midpoint(p1: Point2<f32>, p2: Point2<f32>) -> Point2<f32> {
    fractional_point_on_line(p1, p2, 0.5)
}

fn target_point(from: Point2<f32>, towards: Point2<f32>) -> Point2<f32> {
    let mid = midpoint(from, towards);
    let dist = distance(&from, &towards);
    mid + (towards - from).normalize() * dist * 0.75
}

impl BoundingBox {

    fn new_force_point(&self, mid_start: i8, mid_end: i8, targ_start: i8, targ_end: i8) -> SingleForcePoint {
        SingleForcePoint {
            on_body: fractional_point_on_line(self[mid_start], self[mid_end], 0.2),
            around_joint: target_point(self[targ_start], self[targ_end])
        }
    }

    fn upper_horizontal_forces(&self) -> (SingleForcePoint, SingleForcePoint) {
        (
            self.new_force_point(0, 1, 3,0), // E, A
            self.new_force_point(1, 0, 2,1)  // F, B
        )
    }

    fn right_vertical_forces(&self) -> (SingleForcePoint, SingleForcePoint) {
        (
            self.new_force_point(1,2, 0,1), // E, A
            self.new_force_point(2,1,3,2) // F, B
        )
    }

    fn lower_horizontal_forces(&self) -> (SingleForcePoint, SingleForcePoint) {
        (
            self.new_force_point(3,2,0,3), // H, D
            self.new_force_point(2,3,1,2) // G, C
        )
    }

    fn left_vertical_forces(&self) -> (SingleForcePoint, SingleForcePoint) {
        (
            self.new_force_point(0,3,1,0), // H, D
            self.new_force_point(3,0,2,3) // G, C
        )
    }


    fn get_force_points<H,V>(&self, horizontal:H, vertical: V) -> (SingleForcePoint, SingleForcePoint)
    where
        V: Fn(&Self) -> (SingleForcePoint, SingleForcePoint),
        H: Fn(&Self) -> (SingleForcePoint, SingleForcePoint),

    {
        let side_a_len = distance(&self[0], &self[1]);
        let side_b_len = distance(&self[1], &self[2]);
        if side_a_len < side_b_len {
            vertical(self)
        } else {
            horizontal(self)
        }
    }

    fn force_points(&self) -> ForcePoints {
        let (top_backward, top_forward) = self.get_force_points(Self::upper_horizontal_forces, Self::right_vertical_forces);
        let (bottom_backward, bottom_forward) = self.get_force_points(Self::lower_horizontal_forces, Self::left_vertical_forces);
        ForcePoints {
            top_backward,
            top_forward,
            bottom_backward,
            bottom_forward,
        }
    }
}


#[derive(Copy, Clone, Debug)]
pub struct ModelBody {
    rb: RigidBodyHandle,
    starting_centre: Point2<f32>,
    bounding_box: BoundingBox,
    force_points: ForcePoints,
    join_type: Option<JoinType>,
    max_force_scale: f32,
}

impl ModelBody {

    pub fn  current_centre(&self, rigid_body_set: &RigidBodySet) -> Point2<f32> {
        rigid_body_set[self.rb].position().translation.vector.into()
    }

    pub fn get_bounding_box(&self, rigid_body_set: &RigidBodySet) -> [Point2<f32>; 4] {
        let body_transform = &rigid_body_set[self.rb].position();
        self.bounding_box.0.iter().map(|p| *body_transform * *p).collect::<Vec<_>>().try_into().unwrap()
    }

    pub fn get_far_side_centre(&self, rigid_body_set: &RigidBodySet) -> Point2<f32> {
        let body_transform = rigid_body_set[self.rb].position();
        body_transform*point!(self.bounding_box[1].x, (self.bounding_box[1].y+self.bounding_box[2].y)/2.)
    }

    fn create_body_with_builders(body_set: &mut RigidBodySet,
                                 centre_x: f32,
                                 centre_y: f32,
                                 rbb: RigidBodyBuilder,
                                 collider_set: &mut ColliderSet,
                                 width: f32,
                                 height: f32,
                                 cb: ColliderBuilder,
                                 max_force_scale: f32,
    ) -> Self {
        let body_handle =body_set.insert(rbb.translation(vector![centre_x, centre_y]).angular_damping(2.).build());
        let collider_handle = cb
            .restitution(0.7)
            .friction(0.3)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .build();
        collider_set.insert_with_parent(collider_handle, body_handle, body_set);
        let bounding_box:BoundingBox = [
            point!(-width, height),
            point!(width, height),
            point!(width, -height),
            point!(-width, -height),
        ].into();
        Self {
            rb: body_handle,
            force_points: bounding_box.force_points(),
            starting_centre: point![centre_x, centre_y],
            join_type: None,
            bounding_box,
            max_force_scale
        }
    }

    fn create_dynamic_and_collider(
        body_set: &mut RigidBodySet,
        centre_x: f32,
        centre_y: f32,
        collider_set: &mut ColliderSet,
        width: f32,
        height: f32,
        cb: ColliderBuilder,
        max_force_scale: f32,
    ) -> Self {
        Self::create_body_with_builders(body_set, centre_x, centre_y, RigidBodyBuilder::dynamic()
            .can_sleep(false)
            .ccd_enabled(true), collider_set, width, height, cb, max_force_scale)
    }

    fn create_body_and_collider(
        body_set: &mut RigidBodySet,
        centre_x: f32,
        centre_y: f32,
        collider_set: &mut ColliderSet,
        width: f32,
        height: f32,
        max_force_scale: f32,
    ) -> Self {
        let (cb, jt) = if width>=height {
            (ColliderBuilder::capsule_x(width-height, height), Some(HorizontalJoin))
        } else {
            (ColliderBuilder::capsule_y(height-width, width), Some(VerticalJoin))
        };
        let mut result =
            Self::create_dynamic_and_collider(body_set,centre_x,centre_y,collider_set,width,height, cb, max_force_scale);
        result.join_type = jt;
        result
    }

    fn create_joined_body_and_collider(
        &self,
        join: JoinType,
        body_set: &mut RigidBodySet,
        collider_set: &mut ColliderSet,
        width: f32,
        height: f32, impulse_joint_set: &mut ImpulseJointSet,
        max_force_scale: f32,
    ) -> Self {
        let own_bb = self.get_bounding_box(body_set);
        let own_centre = self.current_centre(body_set);
        let (centre_x, centre_y) = if join == HorizontalJoin {
            (own_bb[1].x+width, own_centre.y)
        } else {
            (own_centre.x, own_bb[2].y-height)
        };
        let follower = Self::create_body_and_collider(body_set, centre_x, centre_y, collider_set, width, height, max_force_scale);
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

    pub(super) fn long_axis_farthest_corner(&self, rigid_body_set: &RigidBodySet) -> Corners {
        let bb = self.get_bounding_box(rigid_body_set);
        if distance(&bb[0],&bb[1])> distance(&bb[1], &bb[2]) {
            ((bb[1].x, bb[1].y), (bb[2].x, bb[2].y))
        } else {
            ((bb[2].x, bb[2].y), (bb[3].x, bb[3].y))
        }
    }

    fn get_pull_force(&self, rigid_body_set: &RigidBodySet, body_rel: SingleForcePoint) -> SingleForcePoint {
        body_rel.transform(rigid_body_set[self.rb].position())
    }

    fn apply_force<T,B>(&self, rigid_body_set: &mut RigidBodySet, force: AdjustedForce, top_provider: T, bottom_provider: B)
    where T: Fn(&Self) -> SingleForcePoint,
          B: Fn(&Self) -> SingleForcePoint,
    {
        let force_point = if force.is_upper() {
            self.get_pull_force(rigid_body_set, top_provider(self))
        } else {
            self.get_pull_force(rigid_body_set, bottom_provider(self))
        };
        // println!("force point: {:?}", force_point);
        let force_vector = force_point.scaled_force_vector(force);
        // println!("force vector: {:?}", force_vector);
        rigid_body_set[self.rb].add_force_at_point(force_vector, force_point.on_body, true);
    }


    fn apply_forward_force(&self, rigid_body_set: &mut RigidBodySet, force: AdjustedForce) {
        self.apply_force(rigid_body_set, force, |s| s.force_points.top_forward, |s| s.force_points.bottom_forward);
    }

    fn apply_backward_force(&self, rigid_body_set: &mut RigidBodySet, force: AdjustedForce) {
        self.apply_force(rigid_body_set, force, |s| s.force_points.top_backward, |s| s.force_points.bottom_backward);
    }


    pub(super) fn apply_force_between(forward:&Self, backward:&Self, rigid_body_set: &mut RigidBodySet, scale: f32) {
        let force_scale = ForceScale::between(forward, backward, scale, rigid_body_set);
        // println!("force scale: {:?}", force_scale);
        forward.apply_forward_force(rigid_body_set, force_scale);
        backward.apply_backward_force(rigid_body_set, force_scale);
    }

    pub fn snapshot(&self, rigid_body_set: &RigidBodySet) -> BodyStateSnapshot {
        let body = &rigid_body_set[self.rb];
        let position = body.position().clone();
        let linear_velocity = body.linvel().clone();
        let angular_velocity = body.angvel();
        BodyStateSnapshot {
            position,
            linear_velocity,
            angular_velocity,
            rb: self.rb,
        }
    }
}


#[cfg(test)]
mod test {
    use rapier2d::dynamics::{CCDSolver, ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet, RevoluteJointBuilder, RigidBodyBuilder, RigidBodySet};
    use rapier2d::geometry::{ColliderBuilder, ColliderSet, DefaultBroadPhase, NarrowPhase};
    use rapier2d::na::{distance, point, vector, Complex, Isometry2, Unit, UnitComplex};
    use rapier2d::pipeline::{ActiveEvents, PhysicsPipeline};
    use crate::physics::modelbody::{BodyStateSnapshot, BoundingBox, ForcePoints, ModelBody, SingleForcePoint, WorldSets};
    use rapier2d::prelude::nalgebra;
    use crate::physics::arm::{TRICEP_HALF_HEIGHT, TRICEP_HALF_WIDTH, TRICEP_MAX_FORCE};
    use crate::physics::modelbody::JoinType::{HorizontalJoin, VerticalJoin};
    use crate::physics::world::{Hangman, PhysicsContext, GROUND_HALF_HEIGHT, WALL_HALF_HEIGHT, WALL_HALF_WIDTH};

    #[test]
    fn test_bounding_box_horizontal() {
        let bounding_box = BoundingBox([point![1.,1.],point![3.,1.], point![3.,0.], point![1.,0.]]);
        let force_points = bounding_box.force_points();
        let expected_force_point = ForcePoints {
            top_backward: SingleForcePoint {
                on_body: point![1.4, 1.],
                around_joint: point![1.,1.25]
            },
            bottom_backward: SingleForcePoint {
                on_body: point![1.4, 0.],
                around_joint: point![1.,-0.25]
            },
            top_forward: SingleForcePoint {
                on_body: point![2.6, 1.],
                around_joint: point![3.,1.25]
            },
            bottom_forward: SingleForcePoint {
                on_body: point![2.6, 0.],
                around_joint: point![3.,-0.25]
            },
        };
        assert_eq!(expected_force_point, force_points);
    }

    #[test]
    fn test_bounding_box_vertical() {
        let bounding_box = BoundingBox([point![1.,3.],point![2.,3.], point![2.,1.], point![1.,1.]]);
        let force_points = bounding_box.force_points();
        let expected_force_point = ForcePoints {
            top_backward: SingleForcePoint {
                on_body: point![2., 2.6],
                around_joint: point![2.25, 3.]
            },
            bottom_backward: SingleForcePoint {
                on_body: point![1., 2.6],
                around_joint: point![0.75, 3.]
            },
            top_forward: SingleForcePoint {
                on_body: point![2., 1.4],
                around_joint: point![2.25, 1.]
            },
            bottom_forward: SingleForcePoint {
                on_body: point![1., 1.4],
                around_joint: point![0.75, 1.]
            },
        };
        assert_eq!(expected_force_point, force_points);
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
            2.
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
        let half_width = 0.25;
        let half_height = 0.04;
        let wall_width = 0.2;
        let wall = ModelBody::create_body_with_builders(
            &mut rigid_body_set, 0.0, 0.1, RigidBodyBuilder::fixed(),
            &mut collider_set, wall_width, 2.0, ColliderBuilder::cuboid(wall_width, 2.0), 0.
        );

        let body_mb = wall.create_joined_body_and_collider(
            HorizontalJoin,
            &mut rigid_body_set,
            &mut collider_set,
            half_width,
            half_height,
            &mut impulse_joint_set,
            2.
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

    #[test]
    fn test_crossing_wall() {
        let mut world_sets = WorldSets::default();
        let hangman = Hangman::new(&mut world_sets);
        let body_mb = world_sets.create_joined_body_and_collider(&hangman.shoulder,
            HorizontalJoin,
            TRICEP_HALF_WIDTH,
            TRICEP_HALF_HEIGHT,
            TRICEP_MAX_FORCE
        );
        let mut context = PhysicsContext::new();
        let mut prev_pos = Vec::new();
        let mut prev_status = body_mb.snapshot(&world_sets.rigid_body_set);
        let wall_dims = hangman.wall.get_bounding_box(&world_sets.rigid_body_set);
        let mut iters = 0;

        loop {
            let force = rand::random_range(-1.0..1.);
            ModelBody::apply_force_between(&hangman.shoulder, &body_mb, &mut world_sets.rigid_body_set, force);
            context.step(&mut world_sets);
            let curr_pos = vec![hangman.wall.get_bounding_box(&world_sets.rigid_body_set), hangman.shoulder.get_bounding_box(&world_sets.rigid_body_set), body_mb.get_bounding_box(&world_sets.rigid_body_set)];
            let curr_status = body_mb.snapshot(&world_sets.rigid_body_set);
            let pos = world_sets.rigid_body_set[body_mb.rb].position();
            let up_right = pos * point![body_mb.bounding_box[1].x-TRICEP_HALF_HEIGHT, body_mb.bounding_box[1].y];
            let down_right = pos * point![body_mb.bounding_box[1].x-TRICEP_HALF_HEIGHT, body_mb.bounding_box[2].y];
            if up_right.x<wall_dims[1].x || down_right.x<wall_dims[1].x {
                println!("{:?}", prev_status);
                println!("{:?}", prev_pos);
                println!("{:?}", curr_pos);
                panic!("body crossed wall {iters}");
            }
            prev_pos = curr_pos;
            prev_status = curr_status;
            iters+=1;
        }
    }

    #[test]
    fn test_dropping_arm() {
        let mut world_sets = WorldSets::default();
        let hangman = Hangman::new(&mut world_sets);
        let body_mb = world_sets.create_joined_body_and_collider(&hangman.shoulder,
                                                                 HorizontalJoin,
                                                                 TRICEP_HALF_WIDTH,
                                                                 TRICEP_HALF_HEIGHT,
                                                                 TRICEP_MAX_FORCE
        );
        let curr_pos = vec![hangman.wall.get_bounding_box(&world_sets.rigid_body_set), hangman.shoulder.get_bounding_box(&world_sets.rigid_body_set), body_mb.get_bounding_box(&world_sets.rigid_body_set)];
        println!("{:?}", curr_pos);
        let a = world_sets.rigid_body_set[body_mb.rb].position();
        let b = a * body_mb.force_points.bottom_backward.around_joint;
        println!("{b}");
        ModelBody::apply_force_between(&hangman.shoulder, &body_mb, &mut world_sets.rigid_body_set, -1.);
    }


    // force point: SingleForcePoint { on_body: [0.14497983, -1.2431198], around_joint: [0.15682957, -1.3067999] }
    // force vector: [[-0.80396146, 4.3204613]]
    // BodyStateSnapshot { rb: RigidBodyHandle(Index { index: 3, generation: 0 }), position: Isometry { rotation: Complex { re: 0.109477155, im: 0.9939893 }, translation: [0.11788662, -1.1465734] }, linear_velocity: [[-4.300039, 1.3039443]], angular_velocity: 30.338573 }
    // [[[-0.065, -0.6999999], [0.065, -0.6999999], [0.065, -1.9], [-0.065, -1.9]], [[0.027499996, -1.2624999], [0.1025, -1.2624999], [0.1025, -1.3375], [0.027499996, -1.3375]], [[0.06364306, -1.2965363], [0.09758098, -0.9883997], [0.17213017, -0.99661046], [0.13819225, -1.3047471]]]
    // [[[-0.065, -0.6999999], [0.065, -0.6999999], [0.065, -1.9], [-0.065, -1.9]], [[0.027499996, -1.2624999], [0.1025, -1.2624999], [0.1025, -1.3375], [0.027499996, -1.3375]], [[0.063563615, -1.301192], [0.05979499, -0.991215], [0.13478945, -0.9903032], [0.13855807, -1.3002803]]]
    #[test]
    fn a_wall_crossing() {
        let mut world_sets = WorldSets::default();
        let collider_customise = |cb:ColliderBuilder| cb.restitution(0.7)
            .friction(0.3)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .build();
        let wall_y = -1.3;
        let wall_h_width = 2.;
        let wall = {
            let body_handle = world_sets.rigid_body_set.insert(RigidBodyBuilder::fixed().translation(vector![-1.935, wall_y]).build());
            let collider_handle = collider_customise(ColliderBuilder::cuboid(wall_h_width, 0.6));
            world_sets.collider_set.insert_with_parent(collider_handle, body_handle, &mut world_sets.rigid_body_set);
            body_handle
        };

        let wall_translation = world_sets.rigid_body_set[wall].position();
        let wall_far_side_centre = wall_translation * point![wall_h_width, 0.];
        let ball_radius = 0.0375;

        let shoulder = {
            let body_handle = world_sets.rigid_body_set.insert(RigidBodyBuilder::fixed().translation(vector![wall_far_side_centre.x, wall_far_side_centre.y]).build());
            let collider_handle = collider_customise(ColliderBuilder::ball(ball_radius));
            world_sets.collider_set.insert_with_parent(collider_handle, body_handle, &mut world_sets.rigid_body_set);
            body_handle
        };

        let shoulder_translation = world_sets.rigid_body_set[shoulder].position();
        let shoulder_far_side_centre = shoulder_translation * point![ball_radius, 0.];

        let half_width = 0.155;
        let new_centre = point![shoulder_far_side_centre.x+half_width, shoulder_far_side_centre.y];

        let attachment = {
            let body_handle =world_sets.rigid_body_set.insert(RigidBodyBuilder::dynamic()
                .can_sleep(false)
                .ccd_enabled(true).translation(vector![new_centre.x, new_centre.y]).build());
            let collider_handle = collider_customise(ColliderBuilder::capsule_x(half_width-ball_radius, ball_radius));
            world_sets.collider_set.insert_with_parent(collider_handle, body_handle, &mut world_sets.rigid_body_set);

            let joint = RevoluteJointBuilder::new()
                .local_anchor1(point![ball_radius, 0.0])
                .local_anchor2(point![-half_width, 0.0])
                .build();

            world_sets.impulse_joint_set.insert(shoulder, body_handle, joint, true);


            let body = &mut world_sets.rigid_body_set[body_handle];

            body.set_position(Isometry2 {
                rotation: Unit::new_normalize(Complex::new(0.109477155, 0.9939893)),
                translation: [0.11788662, -1.1465734].into()
            }, true);
            body.set_linvel(vector![-4.300039, 1.3039443], true);
            body.set_angvel(30.338573, true);

            body_handle
        };

        let mut integration_parameters = IntegrationParameters::default();
        integration_parameters.dt = 1.0 / 240.0;
        integration_parameters.max_ccd_substeps = 4;
        let mut physics_pipeline = PhysicsPipeline::new();
        let mut island_manager = IslandManager::new();
        let mut broad_phase = DefaultBroadPhase::new();
        let mut narrow_phase = NarrowPhase::new();
        let mut ccd_solver = CCDSolver::new();
        let gravity = vector![0.0, -9.81];
        world_sets.rigid_body_set[attachment].add_force_at_point(vector![-0.80396146, 4.3204613], point![0.14497983, -1.2431198], true);
        physics_pipeline.step(
            &gravity, &integration_parameters, &mut island_manager, &mut broad_phase,
            &mut narrow_phase, &mut world_sets.rigid_body_set, &mut world_sets.collider_set,
            &mut world_sets.impulse_joint_set, &mut world_sets.multibody_joint_set, &mut ccd_solver,
            &(), &(),
        );
        let attachment_translation = world_sets.rigid_body_set[attachment].position();
        let attachment_top_right = attachment_translation * point![half_width-ball_radius, ball_radius];
        assert!(attachment_top_right.x > wall_far_side_centre.x);
    }
}