module GPUCode

using oneAPI
using Images

using ..Objects

struct GPUPoint
    x::Float32
    y::Float32
    z::Float32

    GPUPoint(x, y, z) = new(x, y, z)
    GPUPoint(p::Point) = new(p.x, p.y, p.z)
end

struct GPURotation
    pitch::Float32
    yaw::Float32
    roll::Float32

    GPURotation(pitch, yaw, roll) = new(pitch, yaw, roll)
    GPURotation(r::Rotation) = new(r.pitch, r.yaw, r.roll)
end

function calculate_barycentric_coords(inv_denom::Float32, x::Int32, y::Int32, v1::GPUPoint, v2::GPUPoint, v3::GPUPoint)
    w1 = ((v2.y - v3.y) * (x - v3.x) + (v3.x - v2.x) * (y - v3.y)) * inv_denom
    w2 = ((v3.y - v1.y) * (x - v3.x) + (v1.x - v3.x) * (y - v3.y)) * inv_denom
    w3 = 1.0f0 - w1 - w2
    return w1, w2, w3
end

function normalize(v)
    len = sqrt(v[1]*v[1] + v[2]*v[2] + v[3]*v[3])
    if len == 0
        return (0.0f0, 0.0f0, 0.0f0)
    end
    return (v[1]/len, v[2]/len, v[3]/len)
end

function calculate_brightness(n1::Float32, n2::Float32, n3::Float32, light_direction::GPURotation)
    light_len = sqrt(light_direction.pitch^2 + light_direction.yaw^2 + light_direction.roll^2)

    light_norm = (light_direction.pitch / light_len,
                  light_direction.yaw / light_len,
                  light_direction.roll / light_len)

    # normal should already be normalized
    normal_len = sqrt(n1^2 + n2^2 + n3^2)
    if normal_len == 0
        return 0.0f0
    end
    normal_norm = (n1/normal_len, n2/normal_len, n3/normal_len)

    brightness = normal_norm[1] * light_norm[1] +
                 normal_norm[2] * light_norm[2] +
                 normal_norm[3] * light_norm[3]
    return max(brightness, 0.0f0)
end

function fill_triangle_gouraud_gpu(
    S, zbuffer,
    p1::GPUPoint, p2::GPUPoint, p3::GPUPoint,
    b1::Float32, b2::Float32, b3::Float32,
    base_color::RGB{Float32},
    min_x::Int32, min_y::Int32,
    box_width::Int32, box_height::Int32,
    inv_denom::Float32
)
    i = get_global_id()

    y_local, x_local = divrem(i, box_width)

    x = round(Int32, min_x + x_local)
    y = round(Int32, min_y + y_local)

    (w1, w2, w3) = calculate_barycentric_coords(inv_denom, x, y, p1, p2, p3)

    if w1 >= 0 && w2 >= 0 && w3 >= 0
        z = w1 * p1.z + w2 * p2.z + w3 * p3.z

        if z <= zbuffer[y, x]
            zbuffer[y, x] = z

            brightness = w1*b1 + w2*b2 + w3*b3

            ambient = 0.f0

            final_brightness = ambient + (1.0f0 - ambient) * brightness

            final_color = RGB(
                base_color.r * final_brightness,
                base_color.g * final_brightness,
                base_color.b * final_brightness
            )

            S[y, x] = final_color
        end
    end

    return nothing
end

function fill_triangle_phong_gpu(
    S, zbuffer,
    p1::GPUPoint, p2::GPUPoint, p3::GPUPoint,
    n1, n2, n3, #I don't bother typing JuliaGPU arrays because they're annoying
    base_color::RGB{Float32},
    min_x::Int32, min_y::Int32,
    box_width::Int32, box_height::Int32,
    inv_denom::Float32,
    light_direction::GPURotation
)
    i = get_global_id()

    y_local, x_local = divrem(i, box_width)

    x = round(Int32, min_x + x_local)
    y = round(Int32, min_y + y_local)

    (w1, w2, w3) = calculate_barycentric_coords(inv_denom, x, y, p1, p2, p3)

    if w1 >= 0 && w2 >= 0 && w3 >= 0
        z = w1 * p1.z + w2 * p2.z + w3 * p3.z

        if z <= zbuffer[y, x]
            zbuffer[y, x] = z

            normal = (
                w1 * n1[1] + w2 * n2[1] + w3 * n3[1],
                w1 * n1[2] + w2 * n2[2] + w3 * n3[2],
                w1 * n1[3] + w2 * n2[3] + w3 * n3[3],
            )
            normal = normalize(normal)

            brightness = calculate_brightness(normal[1], normal[2], normal[3], light_direction)
            ambient = 0.1f0
            final_brightness = ambient + (1.0f0 - ambient) * brightness

            final_color = RGB(
               base_color.r * final_brightness,
               base_color.g * final_brightness,
               base_color.b * final_brightness
            )

            S[y, x] = final_color
        end
    end

    return nothing
end

export GPUPoint, GPURotation
export fill_triangle_gouraud_gpu, fill_triangle_phong_gpu

end #module
