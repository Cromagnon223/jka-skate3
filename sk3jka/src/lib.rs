//! sk3jka: a small C interface around the Skate 3 Rust engine's skate-host
//! bridge, for the Jedi Academy skate mod. Modelled on how
//! chasmlol/2010-rust-rewrite-mashup drives the same bridge for MW2
//! (crates/render_anim/src/skate.rs, Apache-2.0).
//!
//! Coordinates crossing this interface are Jedi Academy map units (z up,
//! roughly an inch per unit). Inside, the engine works in metres, y up.
//!
//! Threading: one worker thread owns the engine session. The game only sends
//! it jobs and reads the latest published pose, so nothing here blocks a frame.

mod rails;

use bevy::math::{Mat3, Mat4, Vec3, Vec4};
use skate_host::bridge::{ControllerTransport, Pose, Session};
use std::ffi::{CStr, c_char};
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, mpsc};

const API_VERSION: u32 = 1;
const SCALE: f32 = 0.0254;

/// Largest |coordinate| accepted from the map, in map units. Jedi Academy's
/// MAX_WORLD_COORD is 65536; anything far beyond that is garbage, and it would
/// also make rail finding's 64-unit grid walk an enormous number of cells.
const MAX_MAP_COORD: f32 = 262_144.;
/// Map-space sliver filter (squared length of the edge cross product, map
/// units^4). Kept from the original so rail finding sees the same triangles.
const MIN_MAP_CROSS_SQ: f32 = 0.001;
/// Smallest |(b-a)x(c-a)| kept in engine space (m^2, twice the area).
const MIN_ENGINE_CROSS: f32 = 1.0e-7;
/// Smallest squared edge length kept in engine space (m^2): the engine takes
/// a refined inverse square root of each edge and rejects non-finite results.
const MIN_ENGINE_EDGE_SQ: f32 = 1.0e-10;
/// Most simulation time one wake-up may catch up on, seconds.
const MAX_CATCH_UP: f32 = 0.15;
/// Hard cap on engine ticks per wake-up, in case the engine asks for a very
/// short period (slow-motion requests change it). 60 Hz needs at most 9.
const MAX_TICKS_PER_WAKE: u32 = 32;
/// How long sk3_frame waits for its step to finish (it normally takes ~1 ms).
const STEP_WAIT_MS: u64 = 12;
/// Physics recoveries allowed within RECOVERY_WINDOW seconds before giving up.
const MAX_RECOVERIES: u32 = 4;
const RECOVERY_WINDOW: f32 = 3.0;

/// Step requests are numbered; the worker publishes the newest number it has
/// finished so sk3_frame can wait for its own step (the game then draws the
/// pose that already includes this frame's controller input).
static STEP_SEQ: AtomicU64 = AtomicU64::new(0);
static STEP_DONE: Mutex<u64> = Mutex::new(0);
static STEP_CV: Condvar = Condvar::new();

fn mark_step_done(seq: u64) {
    let mut done = STEP_DONE.lock().unwrap_or_else(|e| e.into_inner());
    if seq > *done {
        *done = seq;
    }
    drop(done);
    STEP_CV.notify_all();
}

fn to_skate(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.z, -p.y) * SCALE
}
fn from_skate(p: Vec3) -> Vec3 {
    Vec3::new(p.x, -p.z, p.y) / SCALE
}
/// Rotation taking engine directions to map directions.
fn basis() -> Mat4 {
    Mat4::from_cols(Vec4::X, Vec4::Z, -Vec4::Y, Vec4::W)
}
/// An engine-space transform as a map-space one (map units).
fn to_map(m: Mat4) -> Mat4 {
    let b = basis();
    // `b` is a pure axis permutation, so its inverse is its transpose (exact,
    // and much cheaper than a general 4x4 inverse per bone).
    let mut out = b * m * b.transpose();
    out.w_axis = from_skate(m.w_axis.truncate()).extend(1.);
    out
}
/// A transform as 12 floats: x, y, z axis columns, then origin.
fn flatten(m: Mat4) -> [f32; 12] {
    let (x, y, z, w) = (m.x_axis, m.y_axis, m.z_axis, m.w_axis);
    [x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z, w.x, w.y, w.z]
}

fn coord_ok(c: f32) -> bool {
    c.is_finite() && c.abs() <= MAX_MAP_COORD
}

/// The triangle exactly as the engine will receive it, or None if the engine
/// would reject it. Mirrors skate_world::collision_world (the normal must
/// `try_normalize`) and WorldTriangle::from_vertices (finite vertices, finite
/// non-zero edge lengths), with margins, computed on the very same f32 values.
fn engine_triangle(t: &[Vec3; 3]) -> Option<[[f32; 3]; 3]> {
    if !t.iter().all(|p| coord_ok(p.x) && coord_ok(p.y) && coord_ok(p.z)) {
        return None;
    }
    let e: [[f32; 3]; 3] = t.map(|p| to_skate(p).to_array());
    let [a, b, c] = e.map(Vec3::from_array);
    if !(a.is_finite() && b.is_finite() && c.is_finite()) {
        return None;
    }
    let n = (b - a).cross(c - a);
    let len = n.length();
    if !n.is_finite() || !len.is_finite() || len <= MIN_ENGINE_CROSS || n.try_normalize().is_none() {
        return None;
    }
    for edge in [c - a, b - c, a - b] {
        let sq = edge.length_squared();
        if !sq.is_finite() || sq <= MIN_ENGINE_EDGE_SQ {
            return None;
        }
    }
    Some(e)
}

/// Runs `f`, turning a panic into `fallback` so no panic crosses the C ABI.
fn guard<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(fallback)
}

pub const STATUS_IDLE: i32 = 0;
pub const STATUS_LOADING: i32 = 1;
pub const STATUS_READY: i32 = 2;
pub const STATUS_ACTIVE: i32 = 3;
pub const STATUS_ERROR: i32 = -1;

/// Everything the game reads each frame. Plain C layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Sk3State {
    pub status: i32,
    /// skater root (between the feet), map units
    pub origin: [f32; 3],
    /// skater root rotation, map space: columns of the engine's x, y, z axes
    pub root_axis: [[f32; 3]; 3],
    /// board deck velocity, map units per second
    pub velocity: [f32; 3],
    pub cam_valid: i32,
    pub cam_origin: [f32; 3],
    /// Quake view axis: forward, left, up
    pub cam_axis: [[f32; 3]; 3],
    pub cam_fov: f32,
    pub tick: u64,
    pub bone_count: i32,
    pub controller: i32,
    /// engine state name, NUL terminated
    pub state: [u8; 64],
}

// The C side (bg_skate.h sk3State_t) depends on this exact layout.
const _: () = {
    assert!(std::mem::size_of::<Sk3State>() == 200);
    assert!(std::mem::offset_of!(Sk3State, tick) == 120);
};

const DEFAULT_STATE: Sk3State = Sk3State {
    status: STATUS_IDLE,
    origin: [0.; 3],
    root_axis: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    velocity: [0.; 3],
    cam_valid: 0,
    cam_origin: [0.; 3],
    cam_axis: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    cam_fov: 90.,
    tick: 0,
    bone_count: 0,
    controller: -1,
    state: [0; 64],
};

impl Default for Sk3State {
    fn default() -> Self {
        DEFAULT_STATE
    }
}

struct Shared {
    state: Sk3State,
    /// map-space bone transforms, already flattened for sk3_get_bones
    bones: Vec<[f32; 12]>,
    names: Vec<String>,
    error: String,
    jobs: Option<mpsc::Sender<Job>>,
    log: Option<PathBuf>,
    /// Bumped by every sk3_start. A worker only writes while its generation
    /// is current, so a worker left over from the previous map (possibly
    /// still loading) can never overwrite the new map's state.
    generation: u64,
    /// engine tick length in seconds, 0 until loaded
    period: f32,
    /// short on-screen message for the game (respawn point set, ...)
    notice: String,
    notice_seq: u64,
    /// D-pad respawn point in map space: feet x, y, z and Quake yaw (degrees),
    /// so the game can spawn the player there after dying.
    respawn: Option<[f32; 4]>,
}

static SHARED: Mutex<Shared> = Mutex::new(Shared {
    state: DEFAULT_STATE,
    bones: Vec::new(),
    names: Vec::new(),
    error: String::new(),
    jobs: None,
    log: None,
    generation: 0,
    period: 0.,
    notice: String::new(),
    notice_seq: 0,
    respawn: None,
});

fn shared() -> std::sync::MutexGuard<'static, Shared> {
    SHARED.lock().unwrap_or_else(|e| e.into_inner())
}

fn log(msg: &str) {
    let path = shared().log.clone();
    if let Some(path) = path {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{msg}");
        }
    }
}

fn fail(generation: u64, msg: String) {
    log(&format!("ERROR: {msg}"));
    let mut s = shared();
    if s.generation == generation {
        s.state.status = STATUS_ERROR;
        s.error = msg;
    }
}

fn notify(generation: u64, msg: &str) {
    let mut s = shared();
    if s.generation == generation {
        s.notice.clear();
        s.notice.push_str(msg);
        s.notice_seq = s.notice_seq.wrapping_add(1);
    }
}

/// Yaw (radians, engine convention) of a skater root, inverse of the
/// Quat::from_rotation_y(heading) the bridge uses when activating.
fn heading_of(root: &Mat4) -> f32 {
    let x = root.x_axis;
    let h = (-x.z).atan2(x.x);
    if h.is_finite() { h } else { 0. }
}

/// D-pad up on the controller (XInput bit 0).
const DPAD_UP: u16 = 0x0001;
/// Hold this long to set the respawn point; shorter taps return to it.
const RESPAWN_HOLD: f32 = 0.5;

fn set_status(generation: u64, status: i32) {
    let mut s = shared();
    if s.generation == generation {
        s.state.status = status;
    }
}

enum Job {
    Activate([f32; 3], f32),
    Suspend,
    Step(f32, f32, u64),
}

fn pose_ok(p: &Pose) -> bool {
    p.root.is_finite() && p.bones.iter().all(|b| b.is_finite())
}

/// The worker's side of publishing: everything is computed into buffers it
/// owns, then swapped into SHARED in one short critical section, so the game
/// thread never waits on maths and nothing is allocated per publish once the
/// buffers have grown.
struct Publisher {
    generation: u64,
    /// last published state (status aside), built outside the lock
    mirror: Sk3State,
    /// staging buffer; after each swap it holds the previously published one
    bones: Vec<[f32; 12]>,
    /// the names last handed to SHARED
    names: Vec<String>,
}

impl Publisher {
    fn new(generation: u64) -> Self {
        Self {
            generation,
            mirror: DEFAULT_STATE,
            bones: Vec::new(),
            names: Vec::new(),
        }
    }

    /// Publishes `p`, and also `status` when given (in the same lock).
    fn publish(&mut self, mut p: Pose, controller: Option<usize>, period: f32, status: Option<i32>) {
        let b = basis();
        let root = to_map(p.root);
        let m = &mut self.mirror;
        m.origin = root.w_axis.truncate().to_array();
        m.root_axis = [
            root.x_axis.truncate().to_array(),
            root.y_axis.truncate().to_array(),
            root.z_axis.truncate().to_array(),
        ];
        m.velocity = (b.transform_vector3(p.velocity) / SCALE).to_array();
        match p.camera {
            Some((position, axes, fov)) => {
                let axes: Mat3 = axes;
                let forward = b.transform_vector3(axes.z_axis).normalize_or_zero();
                let up = b.transform_vector3(axes.y_axis).normalize_or_zero();
                let left = up.cross(forward).normalize_or_zero();
                m.cam_valid = 1;
                m.cam_origin = from_skate(position).to_array();
                m.cam_axis = [forward.to_array(), left.to_array(), up.to_array()];
                m.cam_fov = fov;
            }
            None => m.cam_valid = 0,
        }
        m.tick = p.tick;
        m.controller = controller.map_or(-1, |c| c as i32);
        let bytes = p.state.as_bytes();
        let n = bytes.len().min(63);
        m.state = [0; 64];
        m.state[..n].copy_from_slice(&bytes[..n]);

        self.bones.clear();
        self.bones.extend(p.bones.iter().map(|&x| flatten(to_map(x))));
        self.mirror.bone_count = self.bones.len() as i32;

        // Names only change with the skeleton, i.e. practically never.
        let mut new_names: Option<Vec<String>> = None;
        if p.names != self.names {
            self.names.clone_from(&p.names);
            new_names = Some(std::mem::take(&mut p.names));
        }

        {
            let mut st = shared();
            if st.generation != self.generation {
                return;
            }
            let status = status.unwrap_or(st.state.status);
            st.state = self.mirror;
            st.state.status = status;
            std::mem::swap(&mut st.bones, &mut self.bones);
            if let Some(names) = new_names.as_mut() {
                std::mem::swap(&mut st.names, names);
            }
            st.period = period;
        }
        // `p` and the replaced name list are freed here, outside the lock.
    }
}

/// Map-space triangles kept for rail finding, and the same triangles as the
/// engine receives them, with everything the engine would reject dropped.
fn sanitize_triangles(raw: &[f32]) -> (Vec<[Vec3; 3]>, Vec<[[f32; 3]; 3]>) {
    let total = raw.len() / 9;
    let mut map = Vec::with_capacity(total);
    let mut engine = Vec::with_capacity(total);
    let (mut slivers, mut invalid) = (0usize, 0usize);
    for c in raw.chunks_exact(9) {
        let t = [
            Vec3::new(c[0], c[1], c[2]),
            Vec3::new(c[3], c[4], c[5]),
            Vec3::new(c[6], c[7], c[8]),
        ];
        if !(t.iter().all(|v| v.is_finite())
            && (t[1] - t[0]).cross(t[2] - t[0]).length_squared() > MIN_MAP_CROSS_SQ)
        {
            slivers += 1;
            continue;
        }
        match engine_triangle(&t) {
            Some(e) => {
                map.push(t);
                engine.push(e);
            }
            None => invalid += 1,
        }
    }
    if slivers + invalid > 0 {
        log(&format!(
            "collision: kept {} of {} triangles, dropped {} non-finite/sliver, {} the engine would reject",
            map.len(),
            total,
            slivers,
            invalid
        ));
    }
    (map, engine)
}

/// Rails in engine space; any rail the engine could reject is dropped whole
/// (splicing out a point would join unrelated parts of it).
fn sanitize_rails(found: Vec<Vec<Vec3>>) -> Vec<Vec<[f32; 3]>> {
    let total = found.len();
    let mut rails: Vec<Vec<[f32; 3]>> = Vec::with_capacity(total);
    for r in found {
        if r.len() < 2 || !r.iter().all(|p| coord_ok(p.x) && coord_ok(p.y) && coord_ok(p.z)) {
            continue;
        }
        let points: Vec<[f32; 3]> = r.iter().map(|&p| to_skate(p).to_array()).collect();
        if points.iter().all(|p| p.iter().all(|c| c.is_finite())) {
            rails.push(points);
        }
    }
    // The engine counts rails in a u16 (rails::find already caps this).
    rails.truncate(u16::MAX as usize);
    if rails.len() != total {
        log(&format!("rails: dropped {} of {} invalid rails", total - rails.len(), total));
    }
    rails
}

fn worker(generation: u64, root: PathBuf, raw: Vec<f32>, jobs: mpsc::Receiver<Job>) -> Result<(), String> {
    let start = std::time::Instant::now();
    let (tris, triangles) = sanitize_triangles(&raw);
    drop(raw);
    let (found, census) = rails::find(&tris);
    log(&format!(
        "map: {} triangles, {} edges, {} lips, {} runs, {} rails ({}ms)",
        tris.len(),
        census.candidates,
        census.lips,
        census.runs,
        census.rails,
        start.elapsed().as_millis()
    ));
    drop(tris);
    let rails = sanitize_rails(found);
    // Kept so the engine can be rebuilt from scratch if its physics ever gets
    // into a state a simple re-activate can't fix.
    let keep_triangles = triangles.clone();
    let keep_rails = rails.clone();
    let mut session = Session::new(&root, triangles, rails, [0.; 3], 0.)?;
    log(&format!("engine loaded in {}ms", start.elapsed().as_millis()));
    {
        let mut st = shared();
        if st.generation == generation {
            st.state.status = STATUS_READY;
            st.period = session.period();
        }
    }

    let mut publisher = Publisher::new(generation);
    let mut transport = ControllerTransport::default();
    let mut accumulated = 0f32;
    let mut active = false;
    let mut logged_names = false;
    let mut heading = 0f32;
    let mut last_good: Option<([f32; 3], f32)> = None;
    let mut recoveries = 0u32;
    let mut recovery_since = std::time::Instant::now();
    // Respawn point: hold D-pad up to set, tap to go back.
    let mut last_root: Option<Mat4> = None;
    let mut respawn: Option<([f32; 3], f32)> = None;
    let mut dpad_hold = 0f32;
    let mut dpad_placed = false;
    // A job taken off the queue while coalescing steps, handled next.
    let mut next: Option<Job> = None;
    loop {
        let job = match next.take() {
            Some(job) => job,
            None => match jobs.recv() {
                Ok(job) => job,
                Err(_) => break, // sender dropped: shut down or restarted
            },
        };
        match job {
            Job::Activate(spawn, yaw) => {
                accumulated = 0.;
                heading = yaw.to_radians() + std::f32::consts::FRAC_PI_2;
                let p = session.activate(to_skate(Vec3::from_array(spawn)).to_array(), heading)?;
                if !pose_ok(&p) {
                    return Err("the skate engine produced an invalid pose".into());
                }
                last_good = Some((p.root.w_axis.truncate().to_array(), heading));
                last_root = Some(p.root);
                recoveries = 0;
                active = true;
                publisher.publish(p, None, session.period(), Some(STATUS_ACTIVE));
                if !logged_names {
                    logged_names = true;
                    let names = &publisher.names;
                    log(&format!("bones ({}): {}", names.len(), names.join(", ")));
                }
                log(&format!("activated at {spawn:?} yaw {yaw}"));
            }
            Job::Suspend => {
                accumulated = 0.;
                active = false;
                session.suspend_input();
                set_status(generation, STATUS_READY);
            }
            Job::Step(mut dt, mut aspect, mut seq) => {
                // If the game outpaced us, fold every step already queued
                // into this one (total time, newest aspect) so latency never
                // builds up. Any other job stops the fold and runs next, so
                // Activate/Suspend keep their order.
                loop {
                    match jobs.try_recv() {
                        Ok(Job::Step(more, a, s)) => {
                            dt += more;
                            aspect = a;
                            seq = seq.max(s);
                        }
                        Ok(other) => {
                            next = Some(other);
                            break;
                        }
                        Err(_) => break,
                    }
                }
                if !active {
                    mark_step_done(seq);
                    continue;
                }
                let frame = transport.poll();
                let controller = frame.controller();
                let dpad_up = frame.buttons() & DPAD_UP != 0;
                drop(frame);
                session.set_aspect_ratio(aspect);
                accumulated = (accumulated + dt).min(MAX_CATCH_UP);
                let mut ticks = 0u32;
                let mut broken: Option<String> = None;
                loop {
                    // The period can change between ticks (slow motion).
                    let period = session.period();
                    if !(period > 0.) || accumulated < period {
                        break;
                    }
                    if ticks >= MAX_TICKS_PER_WAKE {
                        accumulated = 0.;
                        break;
                    }
                    accumulated -= period;
                    // One fresh controller reading per engine tick, like Skate 3
                    // itself. Reading once per game frame let the engine skip
                    // readings (it only keeps the newest) or see none on some
                    // ticks, which made quick flicks - body flips need theirs
                    // inside a short window at takeoff - sometimes not count.
                    session.collect(transport.poll(), period);
                    if let Err(e) = session.advance() {
                        broken = Some(e);
                        break;
                    }
                    ticks += 1;
                }
                if broken.is_none() && ticks > 0 {
                    let p = session.pose();
                    if pose_ok(&p) {
                        last_good = Some((p.root.w_axis.truncate().to_array(), heading));
                        last_root = Some(p.root);
                        publisher.publish(p, controller, session.period(), None);
                    } else {
                        broken = Some("the skate engine produced an invalid pose".into());
                    }
                }
                if let Some(e) = broken {
                    // Physics blew up (e.g. a bad contact). Like a bail/respawn in
                    // Skate 3: put the skater back on the board at the last good
                    // spot instead of shutting the engine down.
                    if recovery_since.elapsed().as_secs_f32() > RECOVERY_WINDOW {
                        recoveries = 0;
                        recovery_since = std::time::Instant::now();
                    }
                    recoveries += 1;
                    log(&format!("physics error ({e}); recovering ({recoveries})"));
                    let Some((mut spot, h)) = last_good else {
                        mark_step_done(seq);
                        return Err(e);
                    };
                    if recoveries > MAX_RECOVERIES {
                        mark_step_done(seq);
                        return Err(e);
                    }
                    spot[1] += 0.05; // a little above the ground
                    accumulated = 0.;
                    let quick = match session.activate(spot, h) {
                        Ok(p) if pose_ok(&p) => Ok(p),
                        Ok(_) => Err("invalid pose".to_string()),
                        Err(e2) => Err(e2),
                    };
                    match quick {
                        Ok(p) => {
                            publisher.publish(p, controller, session.period(), Some(STATUS_ACTIVE));
                        }
                        Err(e2) => {
                            // Last resort: rebuild the whole engine (a few seconds
                            // of "loading") and drop back in at the same spot.
                            log(&format!("quick recovery failed ({e2}); rebuilding the engine"));
                            set_status(generation, STATUS_LOADING);
                            mark_step_done(seq);
                            session = Session::new(&root, keep_triangles.clone(), keep_rails.clone(), [0.; 3], 0.)
                                .map_err(|e3| format!("{e}; rebuild failed: {e3}"))?;
                            let p = session.activate(spot, h).map_err(|e3| format!("{e}; rebuild activate failed: {e3}"))?;
                            if !pose_ok(&p) {
                                return Err(format!("{e}; rebuilt engine produced an invalid pose"));
                            }
                            transport = ControllerTransport::default();
                            publisher.publish(p, controller, session.period(), Some(STATUS_ACTIVE));
                            log("engine rebuilt");
                        }
                    }
                }
                // Respawn point (D-pad up): hold to set it, tap to return.
                if dpad_up {
                    dpad_hold += dt;
                    if !dpad_placed && dpad_hold >= RESPAWN_HOLD {
                        dpad_placed = true;
                        if let Some(r) = last_root {
                            let h = heading_of(&r);
                            respawn = Some((r.w_axis.truncate().to_array(), h));
                            let feet = from_skate(r.w_axis.truncate());
                            let yaw = (h - std::f32::consts::FRAC_PI_2).to_degrees();
                            {
                                let mut s = shared();
                                if s.generation == generation {
                                    s.respawn = Some([feet.x, feet.y, feet.z, yaw]);
                                }
                            }
                            notify(generation, "Respawn point set");
                            log("respawn point set");
                        }
                    }
                } else {
                    if dpad_hold > 0. && !dpad_placed {
                        match respawn {
                            Some((mut spot, h)) => {
                                spot[1] += 0.05;
                                accumulated = 0.;
                                match session.activate(spot, h) {
                                    Ok(p) if pose_ok(&p) => {
                                        heading = h;
                                        last_good = Some((p.root.w_axis.truncate().to_array(), h));
                                        last_root = Some(p.root);
                                        publisher.publish(p, controller, session.period(), Some(STATUS_ACTIVE));
                                        notify(generation, "Back to respawn point");
                                    }
                                    Ok(_) => log("respawn produced an invalid pose; ignored"),
                                    Err(e) => log(&format!("respawn failed: {e}")),
                                }
                            }
                            None => notify(generation, "Hold D-pad up to set a respawn point"),
                        }
                    }
                    dpad_hold = 0.;
                    dpad_placed = false;
                }
                mark_step_done(seq);
            }
        }
    }
    Ok(())
}

unsafe fn cstr(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p) }.to_str().ok().map(str::to_owned)
}

fn send(job: Job) {
    // Clone the sender (a refcount bump) so the lock is not held while sending.
    let tx = shared().jobs.clone();
    if let Some(tx) = tx {
        let _ = tx.send(job);
    }
}

fn panic_message(panic: &(dyn std::any::Any + Send + 'static)) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "unknown panic".into())
}

#[unsafe(no_mangle)]
pub extern "C" fn sk3_api_version() -> u32 {
    API_VERSION
}

/// Start (or restart for a new map) the engine. `root` is the converted
/// Skate 3 `assets` folder; `tris` is `ntris * 9` floats of map triangles,
/// counter-clockwise seen from the open side. Loading happens in the
/// background: poll `sk3_get_state` for READY or ERROR.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk3_start(root: *const c_char, log_path: *const c_char, tris: *const f32, ntris: u32) -> i32 {
    guard(-1, || {
        let Some(root) = (unsafe { cstr(root) }) else { return -1 };
        let log_path = unsafe { cstr(log_path) };
        // Only copy here; validating and converting happen on the worker so
        // the game thread is not held up by a large map.
        let raw: Vec<f32> = if tris.is_null() || ntris == 0 {
            Vec::new()
        } else {
            let Some(len) = (ntris as usize).checked_mul(9) else { return -1 };
            unsafe { std::slice::from_raw_parts(tris, len) }.to_vec()
        };
        let (tx, rx) = mpsc::channel();
        let generation;
        {
            let mut st = shared();
            st.jobs = Some(tx); // dropping the old sender ends the old worker
            st.generation = st.generation.wrapping_add(1);
            generation = st.generation;
            st.state = Sk3State::default();
            st.state.status = STATUS_LOADING;
            st.error.clear();
            st.bones.clear();
            st.names.clear();
            st.period = 0.;
            st.respawn = None; // new map: the old point means nothing here
            st.log = log_path.map(PathBuf::from);
        }
        log(&format!("sk3jka {} starting, assets: {root}", env!("CARGO_PKG_VERSION")));
        let spawned = std::thread::Builder::new()
            .name("sk3jka".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let result = catch_unwind(AssertUnwindSafe(|| worker(generation, PathBuf::from(&root), raw, rx)));
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => fail(generation, e),
                    Err(panic) => {
                        let msg = panic_message(&*panic);
                        fail(generation, format!("skate engine crashed: {msg}"));
                    }
                }
            });
        if let Err(e) = spawned {
            fail(generation, e.to_string());
            return -1;
        }
        0
    })
}

/// Drop onto the board at a map position facing `yaw_deg` (Quake yaw).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk3_activate(origin: *const f32, yaw_deg: f32) {
    if origin.is_null() {
        return;
    }
    let o = unsafe { [*origin, *origin.add(1), *origin.add(2)] };
    guard((), || {
        if !(o.iter().all(|c| coord_ok(*c)) && yaw_deg.is_finite()) {
            log(&format!("ignored activate at invalid {o:?} yaw {yaw_deg}"));
            return;
        }
        send(Job::Activate(o, yaw_deg));
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn sk3_suspend() {
    guard((), || send(Job::Suspend))
}

/// Advance by `dt` seconds of real time. Reads the Xbox controller itself.
#[unsafe(no_mangle)]
pub extern "C" fn sk3_frame(dt: f32, aspect: f32) {
    if dt.is_finite() && dt > 0. {
        guard((), || {
            let seq = STEP_SEQ.fetch_add(1, Ordering::Relaxed) + 1;
            send(Job::Step(dt.min(0.25), aspect, seq));
            // Only wait while actually skating; never stall a loading or
            // broken engine.
            if shared().state.status != STATUS_ACTIVE {
                return;
            }
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(STEP_WAIT_MS);
            let mut done = STEP_DONE.lock().unwrap_or_else(|e| e.into_inner());
            while *done < seq {
                let now = std::time::Instant::now();
                if now >= deadline {
                    break;
                }
                done = match STEP_CV.wait_timeout(done, deadline - now) {
                    Ok((d, _)) => d,
                    Err(e) => e.into_inner().0,
                };
            }
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk3_get_state(out: *mut Sk3State) -> i32 {
    if out.is_null() {
        return STATUS_ERROR;
    }
    // One 200-byte copy under the lock; the write to the caller happens after.
    let Some(st) = guard(None, || Some(shared().state)) else { return STATUS_ERROR };
    unsafe { out.write_unaligned(st) };
    st.status
}

/// Map-space bone transforms, 12 floats each (x, y, z axis columns, then
/// origin). Returns how many bones were written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk3_get_bones(out: *mut f32, max: i32) -> i32 {
    if out.is_null() || max <= 0 {
        return 0;
    }
    guard(0, || {
        let st = shared();
        // sk3jka: once we're not actively skating, hand back no bones so the
        // game animates the player normally again instead of holding the last
        // skate pose (which survived respawns and broke saber animations).
        if st.state.status != STATUS_ACTIVE {
            return 0;
        }
        let n = st.bones.len().min(max as usize);
        let src: &[f32] = st.bones[..n].as_flattened();
        // Pre-flattened by the worker: a straight copy under the lock.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), out, src.len()) };
        n as i32
    })
}

/// Optional (not used by API version 1 callers): the engine's current tick
/// length in milliseconds, or 0 before the engine has loaded.
/// Latest short on-screen message; returns a counter that changes with each
/// new message (0 = none yet), so the game shows each one once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk3_notice(buf: *mut c_char, len: i32) -> u32 {
    guard(0, || {
        let st = shared();
        if !buf.is_null() && len > 0 {
            let bytes = st.notice.as_bytes();
            let n = bytes.len().min(len as usize - 1);
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, n);
                *buf.add(n) = 0;
            }
        }
        st.notice_seq as u32
    })
}

/// Optional: the D-pad respawn point as map feet x, y, z and Quake yaw
/// (degrees) in `out[0..4]`. Returns 1 if one is set, else 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk3_respawn_point(out: *mut f32) -> i32 {
    guard(0, || {
        let Some(rp) = shared().respawn else { return 0 };
        if out.is_null() || !rp.iter().all(|c| c.is_finite()) {
            return 0;
        }
        unsafe { std::ptr::copy_nonoverlapping(rp.as_ptr(), out, 4) };
        1
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn sk3_period_ms() -> f32 {
    guard(0., || shared().period * 1000.)
}

unsafe fn write_str(s: &str, buf: *mut c_char, len: i32) -> i32 {
    if buf.is_null() || len <= 0 {
        return 0;
    }
    let bytes = s.as_bytes();
    let n = bytes.len().min(len as usize - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, n);
        *buf.add(n) = 0;
    }
    n as i32
}

// The two below copy straight from the shared strings under the lock (at most
// `len` bytes) rather than allocating a clone first.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk3_bone_name(index: i32, buf: *mut c_char, len: i32) -> i32 {
    if buf.is_null() || len <= 0 {
        return 0;
    }
    guard(0, || {
        let st = shared();
        let name = st.names.get(index.max(0) as usize).map_or("", String::as_str);
        unsafe { write_str(name, buf, len) }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk3_error(buf: *mut c_char, len: i32) -> i32 {
    if buf.is_null() || len <= 0 {
        return 0;
    }
    guard(0, || {
        let st = shared();
        unsafe { write_str(&st.error, buf, len) }
    })
}
