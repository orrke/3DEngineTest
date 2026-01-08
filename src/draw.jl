
module Draw

using Images
using ImageView
using ..Objects
using ..CameraModule
using ..Lighting

#Function to project a vertex from the 3D space onto the 2D screen
function project_perspective(point::Union{Point, Coordinates, Vertex}, fov::Float64, aspect_ratio::Float64, width::Int, height::Int)
    if point.z == 0
        return Point(point.x, point.y, point.z)
    end

    projected_x = point.x / point.z
    projected_y = point.y / point.z

    fov_scale = tan(fov/2)

    projected_x = projected_x / (fov_scale * aspect_ratio)
    projected_y = projected_y / fov_scale

    screen_x = Int(round((projected_x + 1) * (width / 2)))
    screen_y = Int(round((1 - projected_y) * (height / 2)))

    return Point(screen_x, screen_y, point.z)
end

#Near plan intersection for Z buffering
function intersect_with_plane(inside_vertex, outside_vertex, plane_z::Float64)
    t = (plane_z - inside_vertex.z) / (outside_vertex.z - inside_vertex.z)

    return Point(
        inside_vertex.x + t * (outside_vertex.x - inside_vertex.x),
        inside_vertex.y + t * (outside_vertex.y - inside_vertex.y),
        plane_z
    )
end

#Near plane clipping to avoid divisions by 0, or by negative numbers
function clip_triangle_near_plane(v1::Vertex, v2::Vertex, v3::Vertex, near_plane::Float64=0.1)
    inside = []
    outside = []

    for v in (v1, v2, v3)
        if v.z >= near_plane
            push!(inside, v)
        else
            push!(outside, v)
        end
    end

    num_inside = length(inside)

    if num_inside == 0
        return []
    elseif num_inside == 3
        return [(v1, v2, v3)]
    elseif num_inside == 1
        visible = inside[1]
        clipped1 = outside[1]
        clipped2 = outside[2]

        new_v1 = intersect_with_plane(visible, clipped1, near_plane)
        new_v2 = intersect_with_plane(visible, clipped2, near_plane)

        return [(visible, new_v1, new_v2)]
    else
        visible1 = inside[1]
        visible2 = inside[2]
        clipped = outside[1]

        new_v1 = intersect_with_plane(visible1, clipped, near_plane)
        new_v2 = intersect_with_plane(visible2, clipped, near_plane)

        return [(visible1, visible2, new_v1), (visible2, new_v1, new_v2)]
    end 
end

#Function to draw a line between 2 vertices, it's mainly used when you have hte wireframe mode enabled
function draw_line(S, p1::Point, p2::Point, thickness::Int=1, depth_buffer=nothing)
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    dz = p2.z - p1.z

    steps = max(abs(dx), abs(dy), abs(dz))

    x_inc = dx / steps
    y_inc = dy / steps
    z_inc = dz / steps

    x = p1.x
    y = p1.y
    z = p1.z

    for _ in 1:steps
        x += x_inc
        y += y_inc
        z += z_inc
        for i in round(Int, x)-round(Int, thickness/2):round(Int, x)+round(Int, thickness/2)
            for j in round(Int, y)-round(Int, thickness/2):round(Int, y)+round(Int, thickness/2)
                if i > 1 && i < size(S, 1) && j > 1 && j < size(S, 2)
                    if isnothing(depth_buffer) || (depth_buffer[j, i] > z)
                        S[j, i] = RGB(255.0, 0.0, 0.0)
                        if !isnothing(depth_buffer)
                            depth_buffer[j, i] = z
                        end
                    end
                end # we want to avoid indexing errors, and still draw triangles that are partially off-screen
            end
        end
    end
end

#Draws a flat triangle on the screen using scanline algorithm
function draw_triangle(S, p1::Point, p2::Point, p3::Point, color::RGB{Float64}, depth_buffer=nothing)
    vertices = sort([p1, p2, p3], by = v -> v.y)
    v1, v2, v3 = vertices[1], vertices[2], vertices[3]

    if v1.y == v3.y
        return
    end

    for y in round(Int, v1.y):round(Int, v3.y)
        if y < 1 || y > size(S, 2)
            continue
        end

        intersections = Vector{Tuple{Float64, Float64}}()

        if v2.y != v1.y && y > v1.y && y <= v2.y
            t = (y - v1.y) / (v2.y - v1.y)
            x = v1.x + t * (v2.x - v1.x)
            z = v1.z + t * (v2.z - v1.z)
            push!(intersections, (x, z))
        end

        if v3.y != v2.y && y >= v2.y && y <= v3.y
            t = (y - v2.y) / (v3.y - v2.y)
            x = v2.x + t * (v3.x - v2.x)
            z = v2.z + t * (v3.z - v2.z)
            push!(intersections, (x, z))
        end

        if v3.y != v1.y && y >= v1.y && y <= v3.y
            t = (y - v1.y) / (v3.y - v1.y)
            x = v1.x + t * (v3.x - v1.x)
            z = v1.z + t * (v3.z - v1.z)
            push!(intersections, (x, z))
        end

        if length(intersections) >= 2
            sort!(intersections, by = p -> p[1])
            
            # Get Left (Start) and Right (End) data
            x_start_f, z_start = intersections[1]
            x_end_f,   z_end   = intersections[end] # Use 'end' in case 3 points hit
            
            x_start = round(Int, x_start_f)
            x_end   = round(Int, x_end_f)

            # Prevent division by zero if it's a 1-pixel wide line
            width_x = x_end_f - x_start_f
            if width_x == 0
                width_x = 1.0 
            end

            # Calculate how much Z changes per 1 pixel of X
            z_slope = (z_end - z_start) / width_x
            
            # Start our running Z counter
            current_z = z_start


            for x in x_start:x_end
                if x > 1 && x < size(S, 1)
                    if isnothing(depth_buffer) || (depth_buffer[y, x] > current_z)
                        S[y, x] = color
                        if !isnothing(depth_buffer)
                            depth_buffer[y, x] = current_z
                        end
                    end
                end
                current_z += z_slope
            end
        end
    end
end

#Function used for backface culling
function triangle_is_facing_camera(normal::Vector{Float64}, v1::Vertex)
    view_direction = [-v1.x, -v1.y, -v1.z]
    
    # Dot product: if positive, face is toward camera
    dot_product = normal[1] * view_direction[1] + 
                  normal[2] * view_direction[2] + 
                  normal[3] * view_direction[3]
    
    return dot_product > 0
end

#Function used to draw an entire mesh to the screen. Currently, it's using the phong shading lighting, but you can change it for gouraud, or flat shading if you change the code a bit.
function draw_mesh(S, meshes::Vector{Mesh}, fov::Float64, aspect_ratio::Float64, width::Int, height::Int, main_light, wireframe::Bool=false)
    Z_buffer = fill(Inf, size(S))

    for mesh in meshes
        for vi in 1:length(mesh.vertices)
            mesh.vertices[vi] = translate_vertex_to_world(mesh.vertices[vi], mesh)
        end
        
        calculate_vertex_normals!(mesh)

        for triangle in mesh.triangles
            v1 = mesh.vertices[triangle.v1]
            v2 = mesh.vertices[triangle.v2]
            v3 = mesh.vertices[triangle.v3]

            clipped_triangles = clip_triangle_near_plane(v1, v2, v3)

            for (v1, v2, v3) in clipped_triangles
                p1 = project_perspective(v1, fov, aspect_ratio, width, height)
                p2 = project_perspective(v2, fov, aspect_ratio, width, height)
                p3 = project_perspective(v3, fov, aspect_ratio, width, height)

                if wireframe
                    draw_line(S, p1, p2, 2, Z_buffer)
                    draw_line(S, p2, p3, 2, Z_buffer)
                    draw_line(S, p3, p1, 2, Z_buffer)
                end

                normal = calculate_normal(triangle, mesh.vertices)

                if !isnothing(p1) && !isnothing(p2) && !isnothing(p3) && triangle_is_facing_camera(normal, v1)
                    # brightness = calculate_brightness(normal, main_light.rotation)

                    # ambient = 0.2
                    # final_brightness = ambient + (1.0 - ambient) * brightness

                    # lit_color = RGB(
                    #     clamp(mesh.color.r * final_brightness, 0.0, 1.0),
                    #     clamp(mesh.color.g * final_brightness, 0.0, 1.0),
                    #     clamp(mesh.color.b * final_brightness, 0.0, 1.0)
                    # )

                    # draw_triangle(S, p1, p2, p3, lit_color, Z_buffer)

                    fill_triangle_phong(S, triangle, mesh, p1, p2, p3, Z_buffer, main_light.rotation)
                end
            end
        end
    end
end

export project_perspective, draw_line, draw_mesh

end #module
