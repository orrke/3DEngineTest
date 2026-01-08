
module Lighting

using Images
using ImageView

using ..Objects

struct Light
    rotation::Rotation
end

#Calculate the brightness from a given normal, and a given light direction
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

#Interpolates brightness over the surface of the triangle using Gouraud shading
function fill_triangle_gouraud(S, triangle::Triangle, mesh::Mesh, p1::Point, p2::Point, p3::Point, zbuffer, light_direction)
    brightness1 = calculate_brightness(mesh.vertex_normals[triangle.v1], light_direction)
    brightness2 = calculate_brightness(mesh.vertex_normals[triangle.v2], light_direction)
    brightness3 = calculate_brightness(mesh.vertex_normals[triangle.v3], light_direction)

    base_color = mesh.color

    vertices = sort([(p1, brightness1), (p2, brightness2), (p3, brightness3)], by = v -> v[1].y)
    (v1, b1), (v2, b2), (v3, b3) = vertices
    
    if v1.y == v3.y
        return
    end

    for y in round(Int, v1.y):round(Int, v3.y)
        if y < 1 || y > size(S, 2)
            continue
        end

        intersections = []

        # Check edge v1-v2
        if v2.y != v1.y && y >= v1.y && y < v2.y
            t = (y - v1.y) / (v2.y - v1.y)
            x = v1.x + t * (v2.x - v1.x)
            z = v1.z + t * (v2.z - v1.z)
            brightness = b1 + t * (b2 - b1)  # Interpolate brightness!
            push!(intersections, (x, z, brightness))
        end
        
        # Check edge v2-v3
        if v3.y != v2.y && y >= v2.y && y <= v3.y
            t = (y - v2.y) / (v3.y - v2.y)
            x = v2.x + t * (v3.x - v2.x)
            z = v2.z + t * (v3.z - v2.z)
            brightness = b2 + t * (b3 - b2)
            push!(intersections, (x, z, brightness))
        end
        
        # Check edge v1-v3
        if v3.y != v1.y && y >= v1.y && y <= v3.y
            t = (y - v1.y) / (v3.y - v1.y)
            x = v1.x + t * (v3.x - v1.x)
            z = v1.z + t * (v3.z - v1.z)
            brightness = b1 + t * (b3 - b1)
            push!(intersections, (x, z, brightness))
        end

        if length(intersections) < 2
            continue
        end

        # Sort by x coordinate
        sort!(intersections, by = i -> i[1])
        (x1, z1, br1) = intersections[1]
        (x2, z2, br2) = intersections[2]
        
        x_min = ceil(Int, x1)
        x_max = floor(Int, x2)
        
        # Draw horizontal span, interpolating brightness
        for x in x_min:x_max
            if x < 1 || x > size(S, 2)
                continue
            end
            
            # Interpolate across the span
            if x_max != x_min
                t = (x - x_min) / (x_max - x_min)
                t = clamp(t, 0.0, 1.0)
                z = z1 + t * (z2 - z1)
                brightness = br1 + t * (br2 - br1)  # Interpolate brightness horizontally!
            else
                z = z1
                brightness = br1
            end
            
            # Z-buffer test
            if z <= zbuffer[y, x]
                zbuffer[y, x] = z
                
                ambient = 0.1
                final_brightness = ambient + (1.0 - ambient) * brightness

                # Apply brightness to base color
                final_color = RGB(
                    base_color.r * final_brightness,
                    base_color.g * final_brightness,
                    base_color.b * final_brightness
                )
                
                S[x, y] = final_color
            end
        end
    end
end

#normalize a normal
function normalize(v)
    len = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    if len == 0
        return [0.0, 0.0, 0.0]
    end
    return [v[1]/len, v[2]/len, v[3]/len]
end

#Barycentric coordinates calculation, used for Phong shading since they're independent of position, and don't result in weird alignment issues on the triangles.
function barycentric_coords(p, v1, v2, v3)
    denom = (v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y)

    w1 = ((v2.y - v3.y) * (p.x - v3.x) + (v3.x - v2.x) * (p.y - v3.y)) / denom
    w2 = ((v3.y - v1.y) * (p.x - v3.x) + (v1.x - v3.x) * (p.y - v3.y)) / denom
    w3 = 1.0 - w1 - w2
    return w1, w2, w3
end

#Fills the triangles using the phong shading method, which interpolates normals for each point on the triangle, and then calculates the brightness for that point.
function fill_triangle_phong(S, triangle::Triangle, mesh::Mesh, p1::Point, p2::Point, p3::Point, zbuffer, light_direction)
    n1 = mesh.vertex_normals[triangle.v1]
    n2 = mesh.vertex_normals[triangle.v2]
    n3 = mesh.vertex_normals[triangle.v3]

    vertices = sort([p1, p2, p3], by = v -> v.y)
    
    if vertices[1].y == vertices[3].y
        return
    end

    # Find bounding box
    min_x = floor(Int, min(p1.x, p2.x, p3.x))
    max_x = ceil(Int, max(p1.x, p2.x, p3.x))
    min_y = floor(Int, min(p1.y, p2.y, p3.y))
    max_y = ceil(Int, max(p1.y, p2.y, p3.y))

    base_color = mesh.color

    for y in min_y:max_y
        for x in min_x:max_x
            if x < 1 || x > size(S, 2) || y < 1 || y > size(S, 1)
                continue
            end

            (w1, w2, w3) = barycentric_coords(Point(x, y, 0.0), p1, p2, p3)

            if w1 >= 0 && w2 >= 0 && w3 >= 0
                z = w1 * p1.z + w2 * p2.z + w3 * p3.z

                if z <= zbuffer[y, x]
                    zbuffer[y, x] = z

                    normal = [
                        w1 * n1[1] + w2 * n2[1] + w3 * n3[1],
                        w1 * n1[2] + w2 * n2[2] + w3 * n3[2],
                        w1 * n1[3] + w2 * n2[3] + w3 * n3[3],
                    ]
                    normal = normalize(normal)

                    brightness = calculate_brightness(normal, light_direction)
                    ambient = 0.2
                    final_brightness = ambient + (1.0 - ambient) * brightness

                    final_color = RGB(
                        clamp(base_color.r * final_brightness, 0.0, 1.0),
                        clamp(base_color.g * final_brightness, 0.0, 1.0),
                        clamp(base_color.b * final_brightness, 0.0, 1.0)
                    )

                    S[y, x] = final_color
                end
            end
        end
    end
end

export Light, calculate_brightness, fill_triangle_gouraud, fill_triangle_phong

end # module