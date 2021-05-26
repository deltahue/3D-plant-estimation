
import os

filepath = pathRoot + "organs/"
pathRoot = "../../../"#'../../3D-data/

# append all objects starting with 'house'
with bpy.data.libraries.load(filepath) as (data_from, data_to):
    data_to.objects = [name for name in data_from.objects]

# link them to scene
scene = bpy.context.scene
for obj in data_to.objects:
    if obj is not None:
        scene.objects.link(obj)
scene.save_as_mainfile("./file.blend")
