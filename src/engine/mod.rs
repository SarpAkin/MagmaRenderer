pub mod material;
pub mod mesh;

use glam::Vec3;
use glam::Vec2;

//packs to VK_FORMAT_A2B10G10R10_SNORM_PACK32
pub fn pack_RGB10_A2_snorm(v: Vec3) -> u32 {
    let scaled = (v.abs() * 511.0 + 0.5).min(Vec3::splat(511.0)); //adding 0.5 for rounding

    let r = scaled.x as u32 | ((scaled.x.is_sign_negative() as u32) << 9);
    let g = scaled.y as u32 | ((scaled.y.is_sign_negative() as u32) << 9);
    let b = scaled.z as u32 | ((scaled.z.is_sign_negative() as u32) << 9);

    r | (g << 10) | (b << 20)
}

pub fn pack_RG16_unorm(v:Vec2) -> u32 {
    let scaled = (v *65535.0 +0.5).clamp(Vec2::splat(0.0), Vec2::splat(65535.0));

    let r = scaled.x as u32;
    let g = scaled.y as u32;

    r | (g << 16)
}
