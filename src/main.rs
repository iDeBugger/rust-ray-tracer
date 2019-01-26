extern crate nalgebra as na;
extern crate image;

use std::time::SystemTime;
use na::{Vector3, Matrix3};
use image::{Rgb, ImageBuffer};

const EPSILON: f64 = 0.00001;

const CANVAS_WIDTH: u32 = 1920;
const CANVAS_HEIGHT: u32 = 1080;

const VIEWPORT_WIDTH: f64 = 1.0;
const VIEWPORT_HEIGHT: f64 = CANVAS_HEIGHT as f64 / CANVAS_WIDTH as f64;
const VIEWPORT_DIST: f64 = 1.0;

const MAX_REFLECT_DEPTH: u32 = 10;

const BACKGROUND_COLOR: Rgb<u8> = Rgb {
    data: [255u8, 255u8, 255u8],
};

fn reflect_ray(ray: Vector3<f64>, normal: Vector3<f64>) -> Vector3<f64> {
    2.0 * normal * na::Matrix::dot(&normal, &ray) - &ray
}

trait RgbExt {
    fn mul(&self, multiplicator: f64) -> Rgb<u8>;
    fn add(&self, add: Rgb<u8>) -> Rgb<u8>;
}
impl RgbExt for Rgb<u8> {
    fn add(&self, add: Rgb<u8>) -> Rgb<u8> {
        let r = std::cmp::max(std::cmp::min(self.data[0] + add.data[0], 255), 0);
        let g = std::cmp::max(std::cmp::min(self.data[1] + add.data[1], 255), 0);
        let b = std::cmp::max(std::cmp::min(self.data[2] + add.data[2], 255), 0);
        Rgb([r, g, b])
    }
    fn mul(&self, multiplicator: f64) -> Rgb<u8> {
        let r = std::cmp::max(std::cmp::min((self.data[0] as f64 * multiplicator) as u64, 255), 0) as u8;
        let g = std::cmp::max(std::cmp::min((self.data[1] as f64 * multiplicator) as u64, 255), 0) as u8;
        let b = std::cmp::max(std::cmp::min((self.data[2] as f64 * multiplicator) as u64, 255), 0) as u8;
        Rgb([r, g, b])
    }
}

struct Scene {
    spheres: Vec<Sphere>,
    lights: Vec<Box<Light>>,
    camera_position: Vector3<f64>,
    camera_rotation: [Matrix3<f64>; 3],
}

impl Scene {
    fn render_to_png(&self) {
        let ts = SystemTime::now();
        let mut img_buf = ImageBuffer::new(CANVAS_WIDTH, CANVAS_HEIGHT);
        for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
            let direction = self.to_viewport_vec(x as i32 - CANVAS_WIDTH as i32 / 2, -(y as i32) + CANVAS_HEIGHT as i32 / 2);
            let direction = self.camera_rotation
                .iter()
                .fold(direction, |direction, rotation| rotation * direction);
            let color = self.trace_ray(self.camera_position, direction, 1.0, std::f64::MAX, MAX_REFLECT_DEPTH);
            *pixel = color;
        };
        dbg!(SystemTime::now().duration_since(ts).unwrap());
        img_buf.save("render.png").unwrap();
    }

    fn to_viewport_vec(&self, x: i32, y: i32) -> Vector3<f64> {
        let v_x = x as f64 * VIEWPORT_WIDTH / CANVAS_WIDTH as f64;
        let v_y = y as f64 * VIEWPORT_HEIGHT / CANVAS_HEIGHT as f64;
        let v_z = VIEWPORT_DIST;
        Vector3::new(v_x, v_y, v_z)
    }

    fn find_closest_intersection(&self, from: Vector3<f64>, direction: Vector3<f64>, dist_min: f64, dist_max: f64)
    -> (Option<&Sphere>, f64) {
        let mut closest_dist = std::f64::MAX;
        let mut closest_sphere = None;

        for sphere in &self.spheres {
            let (t1, t2) = sphere.is_ray_intersect(from, direction);
            if t1 > dist_min && t1 < dist_max && t1 < closest_dist {
                closest_dist = t1;
                closest_sphere = Some(sphere);
            }
            if t2 > dist_min && t2 < dist_max && t2 < closest_dist {
                closest_dist = t2;
                closest_sphere = Some(sphere);
            }
        }

        (closest_sphere, closest_dist)
    }

    fn trace_ray(&self, from: Vector3<f64>, direction: Vector3<f64>, dist_min: f64, dist_max: f64, depth: u32) -> Rgb<u8> {
        let (closest_sphere, closest_dist) = self.find_closest_intersection(from, direction, dist_min, dist_max);

        match closest_sphere {
            None => BACKGROUND_COLOR,
            Some(sphere) => {
                let intersect_point = from + closest_dist * direction;
                let raw_normal = intersect_point - sphere.center;
                let normal = raw_normal / na::Matrix::norm(&raw_normal);
                let mut light_intensity = 0.0;
                for light in &self.lights {
                    light_intensity += light.compute_intensity(&self, intersect_point, normal, -direction, sphere.specular);
                };

                let local_color = sphere.color.mul(light_intensity);

                if depth <= 0 || sphere.reflective <= 0.0 {
                    local_color
                } else {
                    let reflected_ray = reflect_ray(-direction, normal);
                    let reflected_color = self.trace_ray(intersect_point, reflected_ray, EPSILON, std::f64::MAX, depth - 1);

                    local_color.mul(1.0 - sphere.reflective).add(reflected_color.mul(sphere.reflective))
                }
            }
        }
    }
}

struct Sphere {
    center: Vector3<f64>,
    radius: f64,
    color: Rgb<u8>,
    specular: i32,
    reflective: f64,
}

impl Sphere {
    fn is_ray_intersect(&self, orig: Vector3<f64>, dir: Vector3<f64>) -> (f64, f64) {
        let oc = orig - self.center;

        let k1 = na::Matrix::dot(&dir, &dir);
        let k2 = 2.0 * na::Matrix::dot(&oc, &dir);
        let k3 = na::Matrix::dot(&oc, &oc) - self.radius * self.radius;

        let discriminant = k2 * k2 - 4.0 * k1 * k3;
        match discriminant {
            d if d < 0.0 => (std::f64::MAX, std::f64::MAX),
            d => {
                let t1 = (-k2 + d.sqrt()) / (2.0 * k1);
                let t2 = (-k2 - d.sqrt()) / (2.0 * k1);
                (t1, t2)
            }
        }
    }
}

trait Light {
    fn compute_light_vector(&self, point: Vector3<f64>) -> Vector3<f64>;
    fn compute_diffuse_intensity(&self, normal: Vector3<f64>, light: Vector3<f64>) -> f64;
    fn compute_specular_intensity(&self, normal: Vector3<f64>, light: Vector3<f64>, camera: Vector3<f64>, specular: i32) -> f64;
    fn is_point_shadowed(&self, scene: &Scene, point: Vector3<f64>, light: Vector3<f64>) -> bool;
    fn compute_intensity(&self, scene: &Scene, point: Vector3<f64>, normal: Vector3<f64>, camera: Vector3<f64>, specular: i32) -> f64 {
        let light = self.compute_light_vector(point);
        
        if self.is_point_shadowed(scene, point, light) {
            0.0
        } else {
            let diffuse_intensity = self.compute_diffuse_intensity(normal, light);
            let specular_intensity = self.compute_specular_intensity(normal, light, camera, specular);

            diffuse_intensity + specular_intensity
        }
    }
}
struct AmbientLight {
    intensity: f64,
}
impl Light for AmbientLight {
    fn compute_light_vector(&self, _point: Vector3<f64>) -> Vector3<f64> {
        Vector3::new(0.0, 0.0, 0.0)
    }
    fn compute_diffuse_intensity(&self, _normal: Vector3<f64>, _light: Vector3<f64>) -> f64 {
        self.intensity
    }
    fn compute_specular_intensity(&self, _normal: Vector3<f64>, _light: Vector3<f64>, _camera: Vector3<f64>, _specular: i32) -> f64 {
        0.0
    }
    fn is_point_shadowed(&self, _scene: &Scene, _point: Vector3<f64>, _light: Vector3<f64>) -> bool {
        false
    }
}
struct PointLight {
    intensity: f64,
    position: Vector3<f64>,
}
impl Light for PointLight {
    fn compute_light_vector(&self, point: Vector3<f64>) -> Vector3<f64> {
        self.position - point
    }
    fn compute_diffuse_intensity(&self, normal: Vector3<f64>, light: Vector3<f64>) -> f64 {
        let n_dot_l = na::Matrix::dot(&normal, &light);
        if n_dot_l > 0.0 {
            self.intensity * n_dot_l / (na::Matrix::norm(&normal) * na::Matrix::norm(&light))
        } else {
            0.0
        }
    }
    fn compute_specular_intensity(&self, normal: Vector3<f64>, light: Vector3<f64>, camera: Vector3<f64>, specular: i32) -> f64 {
        if specular >= 0 {
            let reflect = reflect_ray(light, normal);
            let r_dot_v = na::Matrix::dot(&reflect, &camera);
            if r_dot_v > 0.0 {
                self.intensity * (r_dot_v / (na::Matrix::norm(&reflect) * na::Matrix::norm(&camera))).powi(specular)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    fn is_point_shadowed(&self, scene: &Scene, point: Vector3<f64>, light: Vector3<f64>) -> bool {
        let dist_max = 1.0;
        let (sphere, _) = scene.find_closest_intersection(point, light, EPSILON, dist_max);

        sphere.is_some()
    }
}
struct DirectionalLight {
    intensity: f64,
    direction: Vector3<f64>,
}
impl Light for DirectionalLight {
    fn compute_light_vector(&self, _point: Vector3<f64>) -> Vector3<f64> {
        self.direction
    }
    fn compute_diffuse_intensity(&self, normal: Vector3<f64>, light: Vector3<f64>) -> f64 {
        let n_dot_l = na::Matrix::dot(&normal, &light);
        if n_dot_l > 0.0 {
            self.intensity * n_dot_l / (na::Matrix::norm(&normal) * na::Matrix::norm(&light))
        } else {
            0.0
        }
    }
    fn compute_specular_intensity(&self, normal: Vector3<f64>, light: Vector3<f64>, camera: Vector3<f64>, specular: i32) -> f64 {
        if specular >= 0 {
            let reflect = reflect_ray(light, normal);
            let r_dot_v = na::Matrix::dot(&reflect, &camera);
            if r_dot_v > 0.0 {
                self.intensity * (r_dot_v / (na::Matrix::norm(&reflect) * na::Matrix::norm(&camera))).powi(specular)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    fn is_point_shadowed(&self, scene: &Scene, point: Vector3<f64>, light: Vector3<f64>) -> bool {
        let dist_max = std::f64::MAX;
        let (sphere, _) = scene.find_closest_intersection(point, light, EPSILON, dist_max);

        sphere.is_some()
    }
}

fn main() {
    let sphere_silver = Sphere {
        center: Vector3::new(0.0, -1.0, 3.0),
        radius: 1.0,
        color: Rgb([255, 153, 51]),
        specular: 500,
        reflective: 0.3,
    };
    let sphere_cyan = Sphere {
        center: Vector3::new(2.0, -0.25, 4.0),
        radius: 1.0,
        color: Rgb([0, 204, 255]),
        specular: 10,
        reflective: 0.001,
    };
    let sphere_green = Sphere {
        center: Vector3::new(-1.5, 0.25, 4.0),
        radius: 1.0,
        color: Rgb([51, 204, 51]),
        specular: 10,
        reflective: 0.75,

    };
    let sphere_gray = Sphere {
        center: Vector3::new(0.0, -5001.0, 0.0),
        radius: 5000.0,
        color: Rgb([153, 153, 153]),
        specular: 1000,
        reflective: 0.001,
    };

    let ambient_light = AmbientLight {
        intensity: 0.2,
    };
    let point_light = PointLight {
        intensity: 0.5,
        position: Vector3::new(2.0, 1.0, 0.0),
    };
    let directional_light = DirectionalLight {
        intensity: 0.2,
        direction: Vector3::new(1.0, 4.0, 4.0),
    };

    let x_rotation = 20.0f64.to_radians();
    let y_rotation = -30.0f64.to_radians();
    let z_rotation = 5.0f64.to_radians();
    let scene = Scene {
        spheres: vec![
            sphere_gray,
            sphere_silver,
            sphere_cyan,
            sphere_green,
        ],
        lights: vec![
            Box::new(ambient_light),
            Box::new(point_light),
            Box::new(directional_light),
        ],
        camera_position: Vector3::new(3.5, 2.5, -2.0),
        camera_rotation: [Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, x_rotation.cos(), -x_rotation.sin(),
            0.0, x_rotation.sin(), x_rotation.cos(),
        ), Matrix3::new(
            y_rotation.cos(), 0.0, y_rotation.sin(),
            0.0, 1.0, 0.0,
            -y_rotation.sin(), 0.0, y_rotation.cos(),
        ), Matrix3::new(
            z_rotation.cos(), -z_rotation.sin(), 0.0,
            z_rotation.sin(), z_rotation.cos(), 0.0,
            0.0, 0.0, 1.0,
        )],
    };
    scene.render_to_png();
}
