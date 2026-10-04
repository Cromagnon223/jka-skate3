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
use std::path::{Path, PathBuf};
use std::sync::{Mutex, mpsc};

const API_VERSION: u32 = 1;
const SCALE: f32 = 0.0254;

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
    let mut out = b * m * b.inverse();
    out.w_axis = from_skate(m.w_axis.truncate()).extend(1.);
    out
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

impl Default for Sk3State {
    fn default() -> Self {
        Self {
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
        }
    }
}

struct Shared {
    state: Sk3State,
    /// map-space bone transforms
    bones: Vec<Mat4>,
    names: Vec<String>,
    error: String,
    jobs: Option<mpsc::Sender<Job>>,
    log: Option<PathBuf>,
}

static SHARED: Mutex<Shared> = Mutex::new(Shared {
    state: Sk3State {
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
    },
    bones: Vec::new(),
    names: Vec::new(),
    error: String::new(),
    jobs: None,
    log: None,
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

fn fail(msg: String) {
    log(&format!("ERROR: {msg}"));
    let mut s = shared();
    s.state.status = STATUS_ERROR;
    s.error = msg;
}

enum Job {
    Activate([f32; 3], f32),
    Suspend,
    Step(f32, f32),
}

fn publish(p: &Pose, controller: Option<usize>) {
    let b = basis();
    let root = to_map(p.root);
    let mut st = shared();
    st.state.origin = root.w_axis.truncate().to_array();
    st.state.root_axis = [
        root.x_axis.truncate().to_array(),
        root.y_axis.truncate().to_array(),
        root.z_axis.truncate().to_array(),
    ];
    st.state.velocity = (b.transform_vector3(p.velocity) / SCALE).to_array();
    match p.camera {
        Some((position, axes, fov)) => {
            let axes: Mat3 = axes;
            let forward = b.transform_vector3(axes.z_axis).normalize_or_zero();
            let up = b.transform_vector3(axes.y_axis).normalize_or_zero();
            let left = up.cross(forward).normalize_or_zero();
            st.state.cam_valid = 1;
            st.state.cam_origin = from_skate(position).to_array();
            st.state.cam_axis = [forward.to_array(), left.to_array(), up.to_array()];
            st.state.cam_fov = fov;
        }
        None => st.state.cam_valid = 0,
    }
    st.state.tick = p.tick;
    st.state.controller = controller.map_or(-1, |c| c as i32);
    let bytes = p.state.as_bytes();
    let n = bytes.len().min(63);
    st.state.state = [0; 64];
    st.state.state[..n].copy_from_slice(&bytes[..n]);
    st.bones = p.bones.iter().map(|m| to_map(*m)).collect();
    if st.names.len() != p.names.len() {
        st.names = p.names.clone();
    }
    st.state.bone_count = st.bones.len() as i32;
}

fn worker(root: PathBuf, tris: Vec<[Vec3; 3]>, jobs: mpsc::Receiver<Job>) -> Result<(), String> {
    let start = std::time::Instant::now();
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
    let triangles: Vec<[[f32; 3]; 3]> = tris
        .iter()
        .map(|t| t.map(|p| to_skate(p).to_array()))
        .collect();
    let rails: Vec<Vec<[f32; 3]>> = found
        .into_iter()
        .map(|r| r.into_iter().map(|p| to_skate(p).to_array()).collect())
        .collect();
    let mut session = Session::new(&root, triangles, rails, [0.; 3], 0.)?;
    log(&format!("engine loaded in {}ms", start.elapsed().as_millis()));
    {
        let mut st = shared();
        st.state.status = STATUS_READY;
    }

    let mut transport = ControllerTransport::default();
    let mut accumulated = 0f32;
    let mut active = false;
    let mut logged_names = false;
    while let Ok(job) = jobs.recv() {
        match job {
            Job::Activate(spawn, yaw) => {
                accumulated = 0.;
                let p = session.activate(
                    to_skate(Vec3::from_array(spawn)).to_array(),
                    yaw.to_radians() + std::f32::consts::FRAC_PI_2,
                )?;
                active = true;
                publish(&p, None);
                shared().state.status = STATUS_ACTIVE;
                if !logged_names {
                    logged_names = true;
                    log(&format!("bones ({}): {}", p.names.len(), p.names.join(", ")));
                }
                log(&format!("activated at {spawn:?} yaw {yaw}"));
            }
            Job::Suspend => {
                accumulated = 0.;
                active = false;
                session.suspend_input();
                shared().state.status = STATUS_READY;
            }
            Job::Step(dt, aspect) => {
                if !active {
                    continue;
                }
                // Only the newest step matters if the game outpaced us.
                let frame = transport.poll();
                let controller = frame.controller();
                session.set_aspect_ratio(aspect);
                session.collect(frame, dt);
                accumulated = (accumulated + dt).min(0.15);
                let mut advanced = false;
                while accumulated >= session.period() {
                    accumulated -= session.period();
                    session.advance()?;
                    advanced = true;
                }
                if advanced {
                    let p = session.pose();
                    if !p.root.is_finite() || p.bones.iter().any(|b| !b.is_finite()) {
                        return Err("the skate engine produced an invalid pose".into());
                    }
                    publish(&p, controller);
                }
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
    let tx = shared().jobs.clone();
    if let Some(tx) = tx {
        let _ = tx.send(job);
    }
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
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(root) = (unsafe { cstr(root) }) else { return -1 };
        let log_path = unsafe { cstr(log_path) };
        let tris: Vec<[Vec3; 3]> = if tris.is_null() {
            Vec::new()
        } else {
            let raw = unsafe { std::slice::from_raw_parts(tris, ntris as usize * 9) };
            raw.chunks_exact(9)
                .map(|c| {
                    [
                        Vec3::new(c[0], c[1], c[2]),
                        Vec3::new(c[3], c[4], c[5]),
                        Vec3::new(c[6], c[7], c[8]),
                    ]
                })
                .filter(|t| t.iter().all(|v| v.is_finite()) && (t[1] - t[0]).cross(t[2] - t[0]).length_squared() > 0.001)
                .collect()
        };
        let (tx, rx) = mpsc::channel();
        {
            let mut st = shared();
            st.jobs = Some(tx); // dropping the old sender ends the old worker
            st.state = Sk3State::default();
            st.state.status = STATUS_LOADING;
            st.error.clear();
            st.bones.clear();
            st.names.clear();
            st.log = log_path.map(PathBuf::from);
        }
        log(&format!("sk3jka {} starting, assets: {root}", env!("CARGO_PKG_VERSION")));
        let spawned = std::thread::Builder::new()
            .name("sk3jka".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let result = catch_unwind(AssertUnwindSafe(|| worker(PathBuf::from(&root), tris, rx)));
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => fail(e),
                    Err(panic) => {
                        let msg = panic
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "unknown panic".into());
                        fail(format!("skate engine crashed: {msg}"));
                    }
                }
            });
        if let Err(e) = spawned {
            fail(e.to_string());
            return -1;
        }
        0
    }));
    r.unwrap_or(-1)
}

/// Drop onto the board at a map position facing `yaw_deg` (Quake yaw).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk3_activate(origin: *const f32, yaw_deg: f32) {
    if origin.is_null() {
        return;
    }
    let o = unsafe { std::slice::from_raw_parts(origin, 3) };
    send(Job::Activate([o[0], o[1], o[2]], yaw_deg));
}

#[unsafe(no_mangle)]
pub extern "C" fn sk3_suspend() {
    send(Job::Suspend);
}

/// Advance by `dt` seconds of real time. Reads the Xbox controller itself.
#[unsafe(no_mangle)]
pub extern "C" fn sk3_frame(dt: f32, aspect: f32) {
    if dt.is_finite() && dt > 0. {
        send(Job::Step(dt.min(0.25), aspect));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk3_get_state(out: *mut Sk3State) -> i32 {
    if out.is_null() {
        return STATUS_ERROR;
    }
    let st = shared().state;
    unsafe { *out = st };
    st.status
}

/// Map-space bone transforms, 12 floats each (x, y, z axis columns, then
/// origin). Returns how many bones were written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk3_get_bones(out: *mut f32, max: i32) -> i32 {
    if out.is_null() || max <= 0 {
        return 0;
    }
    let st = shared();
    let n = st.bones.len().min(max as usize);
    let dst = unsafe { std::slice::from_raw_parts_mut(out, n * 12) };
    for (i, m) in st.bones.iter().take(n).enumerate() {
        let c = [m.x_axis, m.y_axis, m.z_axis, m.w_axis];
        for (j, col) in c.iter().enumerate() {
            dst[i * 12 + j * 3] = col.x;
            dst[i * 12 + j * 3 + 1] = col.y;
            dst[i * 12 + j * 3 + 2] = col.z;
        }
    }
    n as i32
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk3_bone_name(index: i32, buf: *mut c_char, len: i32) -> i32 {
    let name = shared().names.get(index.max(0) as usize).cloned().unwrap_or_default();
    unsafe { write_str(&name, buf, len) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sk3_error(buf: *mut c_char, len: i32) -> i32 {
    let e = shared().error.clone();
    unsafe { write_str(&e, buf, len) }
}

#[allow(dead_code)]
fn _assert_paths(_: &Path) {}
