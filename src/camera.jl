
module CameraModule

using ..Objects

struct Camera
    position::Coordinates #Position relative to the world origin

    rotation::Rotation #Rotation on each axes
end

function camera(position::Coordinates, rotation::Rotation)
    return Camera(position, rotation)
end

function world_space_translation(camera::Camera, objects::Vector{Mesh})
    camera_objects = deepcopy(objects)

    for mesh in camera_objects
        mesh.coordinates.x -= camera.position.x
        mesh.coordinates.y -= camera.position.y
        mesh.coordinates.z -= camera.position.z

        mesh.rotation.pitch -= camera.rotation.pitch
        mesh.rotation.yaw -= camera.rotation.yaw
        mesh.rotation.roll -= camera.rotation.roll
    end

    return camera_objects
end

export Camera, camera, world_space_translation

end
