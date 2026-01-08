
module parser
using ..Objects

"""
Parses an obj file and returns a Mesh object.
"""
function parse_obj_file(file_path::String, coordinates::Coordinates, rotation::Rotation, color::RGB=RGB(.8, .8, .8))
    vertices = []
    normals = []
    triangles = []

    open(file_path, "r") do file
        for line in eachline(file)
            data = split(line)

            if data[1] == "v" # vertices
                x = parse(Float64, data[2])
                y = parse(Float64, data[3])
                z = parse(Float64, data[4])
                push!(vertices, Vertex(x, y, z))
            end

            if data[1] == "vn" # normals
                x = parse(Float64, data[2])
                y = parse(Float64, data[3])
                z = parse(Float64, data[4])
                push!(normals, [x, y, z])
            end

            if data[1] == "f" # faces (triangles)
                v1 = parse(Int, split(data[2], "/")[1])
                v2 = parse(Int, split(data[3], "/")[1])
                v3 = parse(Int, split(data[4], "/")[1])
                push!(triangles, Triangle(v1, v2, v3))
            end

            if data[1] == "mtllib" # material libraries, which aren't supported yet, like textures too
                @info "Material libraries not supported yet."
            end
        end
    end

    mesh = Mesh(coordinates, rotation, vertices, triangles, color, normals)

    return mesh
end

end
