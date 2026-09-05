# External cache NPZ schema

## Structure NPZ accepted keys

At least:

```text
vertices | verts | smpl_vertices | pred_vertices
faces | smpl_faces | triangles
```

Optional:

```text
joints | smpl_joints | pred_joints
camera | cam | pred_cam
```

## Hand NPZ accepted keys

Right hand:

```text
right_vertices | right_hand_vertices | verts_right | mano_right_vertices | vertices
right_faces | right_hand_faces | faces_right | mano_right_faces | faces
```

Left hand:

```text
left_vertices | left_hand_vertices | verts_left | mano_left_vertices
left_faces | left_hand_faces | faces_left | mano_left_faces
```

Optional:

```text
joints | hand_joints | mano_joints
```
