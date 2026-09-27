module GPUPreprocess

using oneAPI
using Images

using ..Objects
using ..GPUCode

function calculate_brightness(normal::Vector{Float64}, light_direction::Rotation)
    light_len = sqrt(light_direction.pitch^2 + light_direction.yaw^2 + light_direction.roll^2)

    light_norm = [light_direction.pitch / light_len,
                  light_direction.yaw / light_len,
                  light_direction.roll / light_len]

    # normal should already be normalized
    normal_len = sqrt(normal[1]^2 + normal[2]^2 + normal[3]^2)
    if normal_len == 0
        return 0.0
    end
    normal_norm = [normal[1]/normal_len, normal[2]/normal_len, normal[3]/normal_len]

    brightness = normal_norm[1] * light_norm[1] +
                 normal_norm[2] * light_norm[2] +
                 normal_norm[3] * light_norm[3]
    return max(brightness, 0.0)
end

function prepare_gouraud_fill_single_triangle(S, triangle::Triangle, mesh::Mesh, p1::Point, p2::Point, p3::Point, zbuffer, light_direction)
    brightness1 = Float32(calculate_brightness(mesh.vertex_normals[triangle.v1], light_direction))
    brightness2 = Float32(calculate_brightness(mesh.vertex_normals[triangle.v2], light_direction))
    brightness3 = Float32(calculate_brightness(mesh.vertex_normals[triangle.v3], light_direction))

    base_color = RGB{Float32}(mesh.color)

    v1, v2, v3 = sort([p1, p2, p3], by = v -> v.y)
    
    if v1.y == v3.y
        return
    end

    # Find bounding box
    min_x = Int32(floor(Int, min(v1.x, v2.x, v3.x)))
    max_x = Int32(ceil(Int, max(v1.x, v2.x, v3.x)))
    min_y = Int32(floor(Int, min(v1.y, v2.y, v3.y)))
    max_y = Int32(ceil(Int, max(v1.y, v2.y, v3.y)))

    denom = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y)
    inv_denom = Float32(inv(denom))

    gpu_screen = oneArray(S)
    gpu_zbuffer = oneArray(zbuffer)

    pixel_count = Int(max_x-min_x) * Int(max_y-min_y)
    items = 256
    groups = cld(pixel_count, items)

    @oneapi items=items groups=groups GPUCode.fill_triangle_gouraud_gpu(
        gpu_screen, gpu_zbuffer,
        GPUCode.GPUPoint(p1), GPUCode.GPUPoint(p2), GPUCode.GPUPoint(p3),
        brightness1, brightness2, brightness3,
        base_color,
        min_x, min_y,
        max_x-min_x, max_y-min_y,
        inv_denom
        )

    copyto!(S, gpu_screen)
    copyto!(zbuffer, gpu_zbuffer)
end

function prepare_phong_fill_single_triangle(S, triangle::Triangle, mesh::Mesh, p1::Point, p2::Point, p3::Point, zbuffer, light_direction)
    n1 = oneArray(Float32.(mesh.vertex_normals[triangle.v1]))
    n2 = oneArray(Float32.(mesh.vertex_normals[triangle.v2]))
    n3 = oneArray(Float32.(mesh.vertex_normals[triangle.v3]))

    base_color = RGB{Float32}(mesh.color)

    v1, v2, v3 = sort([p1, p2, p3], by = v -> v.y)
    
    if v1.y == v3.y
        return
    end

    # Find bounding box
    min_x = Int32(floor(Int, min(v1.x, v2.x, v3.x)))
    max_x = Int32(ceil(Int, max(v1.x, v2.x, v3.x)))
    min_y = Int32(floor(Int, min(v1.y, v2.y, v3.y)))
    max_y = Int32(ceil(Int, max(v1.y, v2.y, v3.y)))

    denom = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y)
    inv_denom = Float32(inv(denom))

    gpu_screen = oneArray(S)
    gpu_zbuffer = oneArray(zbuffer)

    pixel_count = Int(max_x-min_x) * Int(max_y-min_y)
    items = 256
    groups = cld(pixel_count, items)

    gpu_light = GPURotation(light_direction)

    @oneapi items=items groups=groups GPUCode.fill_triangle_phong_gpu(
        gpu_screen, gpu_zbuffer,
        GPUCode.GPUPoint(p1), GPUCode.GPUPoint(p2), GPUCode.GPUPoint(p3),
        n1, n2, n3,
        base_color,
        min_x, min_y,
        max_x-min_x, max_y-min_y,
        inv_denom,
        gpu_light
        )

    copyto!(S, gpu_screen)
    copyto!(zbuffer, gpu_zbuffer)
end

export prepare_gouraud_fill_single_triangle, prepare_phong_fill_single_triangle

end