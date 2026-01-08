using Images
using ImageView
using Makie
using GLMakie

include("objects.jl")
include("lighting.jl")
include("camera.jl")
include("draw.jl")

using .Objects
using .CameraModule
using .Draw
using .Lighting

height = 1000
width = 1000

#Create the screen buffer (Framebuffer)
screen = zeros(RGB{Float64}, (height, width))

cube = Objects.create_cube(Objects.Coordinates(0, 0, 0), Objects.Rotation(0, 0, 0), 10)
#plane = Objects.create_plane(Objects.Coordinates(-10, 0, 15), Objects.Rotation(0, 0, 0), 20, 20)

#World space, just has a cube that way it's simple to test
world_space = [cube]

#Main light, currently only one since I haven't implemented multiple lights yet
main_light = Light(Rotation(-1,-1,0))

#The camera
camera = CameraModule.camera(Objects.Coordinates(0, 0, -10), Objects.Rotation(0, 0, 0))

#Translation to camera space
camera_space = CameraModule.world_space_translation(camera, world_space)

#Draw the mesh to the screen
Draw.draw_mesh(screen, camera_space, 90.0, width/height, width, height, main_light)

#Use GLMakie to display the screen buffer
fig = Figure()
ax = GLMakie.Axis(fig[1, 1])
img = image!(ax, screen)

#Slide to rotate the cube, it's for testing the lighting algorithm
slider = Slider(fig[2, 1], range =  0:1:360, startvalue = 5.0)

on(slider.value) do val
    camera.rotation.roll = val

    camera_space = CameraModule.world_space_translation(camera, world_space)
    screen = zeros(RGB{Float64}, (height, width))
    Draw.draw_mesh(screen, camera_space, 90.0, width/height, width, height, main_light)

    img[1] = screen
end

#Display the image onto the screen.
display(fig)
