"""Python bindings for 3D gaussian projection"""

from typing import Tuple

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C


def project_gaussians_2d_scale_rot(
    means2d: Float[Tensor, "*batch 2"],
    scales2d: Float[Tensor, "*batch 2"],
    rotation: Float[Tensor, "*batch 1"],
    img_height: int,
    img_width: int,
    tile_bounds: Tuple[int, int, int],
    clip_thresh: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       projmat (Tensor): projection matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).
       clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
    """
    return _ProjectGaussians2dScaleRot.apply(
        means2d.contiguous(),
        scales2d.contiguous(),
        rotation.contiguous(),
        img_height,
        img_width,
        tile_bounds,
        clip_thresh,
    )

class _ProjectGaussians2dScaleRot(Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means2d: Float[Tensor, "*batch 2"],
        scales2d: Float[Tensor, "*batch 2"],
        rotation: Float[Tensor, "*batch 1"],
        img_height: int,
        img_width: int,
        tile_bounds: Tuple[int, int, int],
        clip_thresh: float = 0.01,
    ):
        num_points = means2d.shape[-2]
        if num_points < 1 or means2d.shape[-1] != 2:
            raise ValueError(f"Invalid shape for means2d: {means2d.shape}")
        (
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
        ) = _C.project_gaussians_2d_scale_rot_forward(
            num_points,
            means2d,
            scales2d,
            rotation,
            img_height,
            img_width,
            tile_bounds,
            clip_thresh,
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points

        # Save tensors.
        ctx.save_for_backward(
            means2d,
            scales2d,
            rotation,
            radii,
            conics,
        )

        return (xys, depths, radii, conics, num_tiles_hit)

    @staticmethod
    def backward(ctx, v_xys, v_depths, v_radii, v_conics, v_num_tiles_hit):
        (
            means2d,
            scales2d,
            rotation,
            radii,
            conics,
        ) = ctx.saved_tensors

        (v_cov2d, v_mean2d, v_scale, v_rot) = _C.project_gaussians_2d_scale_rot_backward(
            ctx.num_points,
            means2d,
            scales2d,
            rotation,
            ctx.img_height,
            ctx.img_width,
            radii,
            conics,
            v_xys,
            v_depths,
            v_conics,
        )

        # Return a gradient for each input.
        return (
            # means2d: Float[Tensor, "*batch 2"],
            v_mean2d,
            # scales: Float[Tensor, "*batch 2"],
            v_scale,
            #rotation: Float,
            v_rot,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # tile_bounds: Tuple[int, int, int],
            None,
            # clip_thresh,
            None,
        )


def project_gaussians_2d_scale_rot_batch(
    batch_size: int,
    num_points_per_image: int,
    means2d: Float[Tensor, "batch_times_n 2"],
    scales2d: Float[Tensor, "batch_times_n 2"],
    rotation: Float[Tensor, "batch_times_n 1"],
    img_height: int,
    img_width: int,
    tile_bounds: Tuple[int, int, int],
    clip_thresh: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(BATCH VERSION) This function projects 3D gaussians to 2D for a batch of images using Scale and Rotation.

    Note:
        This function is differentiable w.r.t the means2d, scales2d and rotation inputs.

    Args:
        batch_size (int): The number of images in the batch.
        num_points_per_image (int): The number of gaussians per image.
        means2d (Tensor): xy locations of 2D gaussian projections for the entire batch.
                          Shape: (batch_size * num_points_per_image, 2)
        scales2d (Tensor): Scaling factors for the 2D gaussians.
                           Shape: (batch_size * num_points_per_image, 2)
        rotation (Tensor): Rotation angle (radians) for the 2D gaussians.
                           Shape: (batch_size * num_points_per_image, 1)
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        tile_bounds (Tuple): tile dimensions for a single image, as a len 3 tuple (tiles.x, tiles.y, 1).
        clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
    """
    return _ProjectGaussians2dScaleRotBatch.apply(
        batch_size,
        num_points_per_image,
        means2d.contiguous(),
        scales2d.contiguous(),
        rotation.contiguous(),
        img_height,
        img_width,
        tile_bounds,
        clip_thresh,
    )


class _ProjectGaussians2dScaleRotBatch(Function):
    """(BATCH VERSION) PyTorch Function for projecting 3D gaussians to 2D using Scale/Rot."""

    @staticmethod
    def forward(
        ctx,
        batch_size: int,
        num_points_per_image: int,
        means2d: Float[Tensor, "batch_times_n 2"],
        scales2d: Float[Tensor, "batch_times_n 2"],
        rotation: Float[Tensor, "batch_times_n 1"],
        img_height: int,
        img_width: int,
        tile_bounds: Tuple[int, int, int],
        clip_thresh: float = 0.01,
    ):
        """The forward pass of the projection."""
        
        # 调用我们在 C++ 中新增的 batch forward 函数
        (
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
        ) = _C.project_gaussians_2d_scale_rot_batch_forward(
            batch_size,
            num_points_per_image,
            means2d,
            scales2d,
            rotation,
            img_height,
            img_width,
            tile_bounds,
            clip_thresh,
        )

        # Save non-tensor context variables
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.batch_size = batch_size
        ctx.num_points_per_image = num_points_per_image

        # Save tensors needed for backward
        ctx.save_for_backward(
            means2d,
            scales2d,
            rotation,
            radii,
            conics,
        )

        return (xys, depths, radii, conics, num_tiles_hit)

    @staticmethod
    def backward(ctx, v_xys, v_depths, v_radii, v_conics, v_num_tiles_hit):
        """The backward pass of the projection."""
        
        (
            means2d,
            scales2d,
            rotation,
            radii,
            conics,
        ) = ctx.saved_tensors

        # 调用我们在 C++ 中新增的 batch backward 函数
        (v_mean2d, v_scale, v_rot) = _C.project_gaussians_2d_scale_rot_batch_backward(
            ctx.batch_size,
            ctx.num_points_per_image,
            means2d,
            scales2d,
            rotation,
            ctx.img_height,
            ctx.img_width,
            radii,
            conics,
            v_xys,
            v_depths,
            v_conics,
        )

        # 返回每个输入参数对应的梯度，不需要梯度的位置返回 None
        return (
            # batch_size
            None,
            # num_points_per_image
            None,
            # means2d
            v_mean2d,
            # scales2d
            v_scale,
            # rotation
            v_rot,
            # img_height
            None,
            # img_width
            None,
            # tile_bounds
            None,
            # clip_thresh
            None,
        )