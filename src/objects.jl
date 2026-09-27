

module Objects

using LinearAlgebra
using Images

#Basic data structures for 3D objects
#Coordinates, Vertex and Point are basicalyl the same, but used in different contexts for clarity.
mutable struct Coordinates
    x::Float64
    y::Float64
    z::Float64

    Coordinates(x, y, z) = new(x, y, z)
end

mutable struct Rotation
    pitch::Float64
    yaw::Float64
    roll::Float64

    Rotation(pitch, yaw, roll) = new(pitch, yaw, roll)
end

mutable struct Point
    x::Float64
    y::Float64
    z::Float64

    Point(x, y, z) = new(x, y, z)
    Point(c::Coordinates) = new(c.x, c.y, c.z)
end

function Base.Tuple(p::Point)
    return (p.x, p.y, p.z)
end

mutable struct Vertex
    x::Float64
    y::Float64
    z::Float64

    Vertex(x, y, z) = new(x, y, z)
    Vertex(p::Point) = new(p.x, p.y, p.z)
    Vertex(c::Coordinates) = new(c.x, c.y, c.z)
end

function Point(vertex::Vertex)
    return Point(vertex.x, vertex.y, vertex.z)
end

#Helper function to add a Vertex and Coordinates together
function Base.:+(v1::Vertex, v2::Coordinates)
    return Vertex(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)
end

#Could also be called faces
struct Triangle
    #vertices are the indices of the vertices in the mesh's vertex list
    v1::Int
    v2::Int
    v3::Int

    Triangle(v1::Int, v2::Int, v3::Int) = new(v1, v2, v3)
end

#Allow indexing into Triangle to get vertex indices
function Base.getindex(triangle::Triangle, index::Int)
    if index == 1
        return triangle.v1
    elseif index == 2
        return triangle.v2
    elseif index == 3
        return triangle.v3
    else
        throw(ArgumentError("Triangle only has indices 1, 2, and 3"))
    end
end

#Calculates the normal vector of a triangle
function calculate_normal(triangle::Triangle, vertices::Vector{Vertex})
    v1 = vertices[triangle.v1]
    v2 = vertices[triangle.v2]
    v3 = vertices[triangle.v3]

    u = [v2.x - v1.x, v2.y - v1.y, v2.z - v1.z]
    v = [v3.x - v1.x, v3.y - v1.y, v3.z - v1.z]

    normal = cross(u, v)

    length = sqrt(normal[1]^2 + normal[2]^2 + normal[3]^2)
    
    if length == 0
        return [0.0, 0.0, 0.0]
    end
    
    return [normal[1]/length, normal[2]/length, normal[3]/length]
end

#Dot product of the normals of two triangles
function normal_dot_product(t1::Triangle, t2::Triangle, vertices::Vector{Vertex})
    n1 = calculate_normal(t1, vertices)
    n2 = calculate_normal(t2, vertices)

    return n1[1]*n2[1] + n1[2]*n2[2] + n1[3]*n2[3]
end

#Mesh structure, holds the essential information related to an object.
mutable struct Mesh
    coordinates::Coordinates
    rotation::Rotation

    vertices::Vector{Vertex}
    triangles::Vector{Triangle}

    color::RGB

    vertex_normals::Vector{Vector{Float64}}

    Mesh(coordinates::Coordinates, rotation::Rotation, vertices::Vector{Vertex}, triangles::Vector{Triangle}) = new(coordinates, rotation, vertices, triangles, RGB(0.8, 0.8, 0.8), [])
end

#Calculates the normal for a given vertex by averaging the normals of all connected triangles
function vertex_normal_calculation(vertex_index::Int, mesh::Mesh)
    connected_triangles = []

    for triangle in mesh.triangles
        if triangle.v1 == vertex_index || triangle.v2 == vertex_index || triangle.v3 == vertex_index
            push!(connected_triangles, triangle)
        end
    end

    normal_sum = [0.0, 0.0, 0.0]

    for triangle in connected_triangles
        normal = calculate_normal(triangle, mesh.vertices)
        normal_sum[1] += normal[1]
        normal_sum[2] += normal[2]
        normal_sum[3] += normal[3]
    end

    normal_sum_len = sqrt(normal_sum[1]^2 + normal_sum[2]^2 + normal_sum[3]^2)
    normal_sum = [normal_sum[1] / normal_sum_len,
                  normal_sum[2] / normal_sum_len,
                  normal_sum[3] / normal_sum_len]

    return normal_sum
end

#Functions to rotate vertices around the x axis, y axis, and z axis respectively
function rotate_x(vertex::Vertex, angle::Float64)
    return Vertex(
        vertex.x,
        vertex.y * cos(angle) - vertex.z * sin(angle),  # NO rounding!
        vertex.y * sin(angle) + vertex.z * cos(angle)   # NO rounding!
    )
end

function rotate_y(vertex::Vertex, angle::Float64)
    return Vertex(
        vertex.x * cos(angle) + vertex.z * sin(angle),
        vertex.y,
        -vertex.x * sin(angle) + vertex.z * cos(angle)
    )
end

function rotate_z(vertex::Vertex, angle::Float64)
    return Vertex(
        vertex.x * cos(angle) - vertex.y * sin(angle),
        vertex.x * sin(angle) + vertex.y * cos(angle),
        vertex.z
    )
end

#Rotates a vertex by given rotation angles (in degrees), with that specific order, because it's the standard
function rotate_vertex(vertex::Vertex, rotation::Rotation)
    rotated = rotate_y(vertex, deg2rad(rotation.yaw))
    rotated = rotate_x(rotated, deg2rad(rotation.pitch))
    rotated = rotate_z(rotated, deg2rad(rotation.roll))
    return rotated
end

#Translates a vertex to world space based on the mesh's coordinates and rotation
function translate_vertex_to_world(vertex::Vertex, mesh::Mesh)
    rotated = rotate_vertex(vertex, mesh.rotation)

    return rotated + mesh.coordinates
end

#Gets the triangle that contains the given two vertex indices
function get_triangle_from_line(mesh::Mesh, i1::Int, i2::Int)
    for triangle in mesh.triangles
        if (triangle.v1 == i1 && triangle.v2 == i2) ||
           (triangle.v2 == i1 && triangle.v3 == i2) ||
           (triangle.v3 == i1 && triangle.v1 == i2) ||
           (triangle.v1 == i2 && triangle.v2 == i1) ||
           (triangle.v2 == i2 && triangle.v3 == i1) ||
           (triangle.v3 == i2 && triangle.v1 == i1)
            return triangle
        end
    end
    return nothing
end

#Calculates and sets the vertex normals for the entire mesh
function calculate_vertex_normals!(mesh::Mesh)
    mesh.vertex_normals = []

    for i in 1:length(mesh.vertices)
        normal = vertex_normal_calculation(i, mesh)
        push!(mesh.vertex_normals, normal)
    end
end

#Creates a cube mesh given an origin, rotation, and size
function create_cube(origin::Coordinates, rotation::Rotation, size::Int)
    half = round(Int, size / 2.0)
    other_half = size - half # to handle odd sizes

    #Top of the cube
    v1 = Vertex(-half, -half, -half) #front top right
    v2 = Vertex(other_half, -half, -half) #front top left
    v3 = Vertex(-half, -half, other_half) #back top right
    v4 = Vertex(other_half, -half, other_half) # back top left
    #Bottom of the cube
    v5 = Vertex(-half, other_half, -half) #back bottom right
    v6 = Vertex(other_half, other_half, -half) #back bottom left
    v7 = Vertex(-half, other_half, other_half) #front bottom right
    v8 = Vertex(other_half, other_half, other_half) # front bottom left


    #0;0;0 is the front top right corner of the cube

    #top of the cube
    t1 = Triangle(2, 3, 1)
    t2 = Triangle(4, 3, 2)
    #left side of the cube
    t3 = Triangle(5, 1, 3)
    t4 = Triangle(7, 5, 3)
    #right side of the cube
    t5 = Triangle(2, 6, 4)
    t6 = Triangle(4, 6, 8)
    #back side
    t7 = Triangle(3, 4, 7)
    t8 = Triangle(8, 7, 4)
    #below
    t9 = Triangle(5, 7, 6)
    t10 = Triangle(6, 7, 8)
    #front
    t11 = Triangle(1, 5, 2)
    t12 = Triangle(5, 6, 2)

    #creating the cube
    vertices = [v1, v2, v3, v4, v5, v6, v7, v8]
    triangles = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12]

    cube = Mesh(origin, rotation, vertices, triangles)
    
    return cube
end

#Creates a plane mesh given an origin, rotation, width, and height
function create_plane(origin::Coordinates, rotation::Rotation, width::Int, height::Int)
    half_width = round(Int, width/2.0)
    half_height = round(Int, height/2.0)

    v1 = Vertex(half_width, half_width, 0)
    v2 = Vertex(-half_width, half_width, 0)
    v3 = Vertex(-half_width, -half_width, 0)
    v4 = Vertex(half_width, -half_width, 0)

    t1 = Triangle(1, 2, 3)
    t2 = Triangle(1, 3, 4)

    vertices = [v1, v2, v3, v4]
    triangles = [t1, t2]

    plane = Mesh(origin, rotation, vertices, triangles)
end

export Coordinates, Rotation, Point, Vertex, Triangle, Mesh, create_cube, rotate_vertex, translate_vertex_to_world, get_triangle_from_line, calculate_normal, calculate_vertex_normals!, create_plane, vertex_normal_calculation

end
