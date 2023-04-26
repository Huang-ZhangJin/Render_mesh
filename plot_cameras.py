import torch
import numpy as np


def lift(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (
        (
            x
            - cx.unsqueeze(-1)
            + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
            - sk.unsqueeze(-1) * y / fy.unsqueeze(-1)
        )
        / fx.unsqueeze(-1)
        * z
    )
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)


def draw_camera(
    pose_all: torch.Tensor,
    intrinsics_all: torch.Tensor,
    save_name: str = "posetest.ply",
    h: int = 800,
    w: int = 800,
) -> None:
    import open3d as o3d  # Note the version: pip install open3d==0.10.0

    ###########################################################
    ########## Code for visualizing cameras ###################
    ###########################################################
    pose_all = pose_all.cuda()
    intrinsics_all = intrinsics_all.cuda()

    uv = np.stack(np.meshgrid([0, h - 1], [0, w - 1], indexing="ij"), axis=-1).reshape(-1, 2)
    uv = torch.from_numpy(uv).type(torch.cuda.FloatTensor)
    uv = uv.unsqueeze(0).expand(pose_all.shape[0], 4, 2)

    batch_size, num_samples, _ = uv.shape
    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    # z_cam = -depth.view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)
    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics_all)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    cam_loc = pose_all[:, :3, 3]  # [n_imgs, 3]
    world_coords = torch.bmm(pose_all, pixel_points_cam).permute(0, 2, 1)[
        :, :, :3
    ]  # [n_imgs, n_pixels, 3]

    points_all = torch.cat([cam_loc, world_coords.reshape(-1, 3)], dim=0).cpu()
    pixel_corner_idxs = (
        torch.arange(num_samples).type(torch.LongTensor).reshape(1, -1)
        + torch.arange(batch_size).type(torch.LongTensor).reshape(-1, 1) * num_samples
    )
    pixel_corner_idxs = pixel_corner_idxs + batch_size
    cam_loc_idxs_exp = (
        torch.arange(batch_size).type(torch.LongTensor).unsqueeze(1).expand(batch_size, num_samples)
    )
    edges = torch.stack([cam_loc_idxs_exp, pixel_corner_idxs], dim=2)
    edges = edges.reshape(-1, 2)
    pixel_corner_idxs_shift = pixel_corner_idxs[:, [1, 3, 0, 2]]
    img_plane_edges = torch.stack([pixel_corner_idxs, pixel_corner_idxs_shift], dim=2)
    img_plane_edges = img_plane_edges.reshape(-1, 2)
    edges = torch.cat([edges, img_plane_edges], dim=0)

    cam_edgeset = o3d.geometry.LineSet()
    cam_edgeset.points = o3d.utility.Vector3dVector(points_all.numpy())
    cam_edgeset.lines = o3d.utility.Vector2iVector(edges.numpy())

    o3d.io.write_line_set(save_name, cam_edgeset)
    print("Saved Pose in PLY format")

