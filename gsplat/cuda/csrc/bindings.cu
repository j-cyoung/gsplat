#include "backward.cuh"
#include "bindings.h"
#include "forward.cuh"
#include "forward2d.cuh"
#include "backward2d.cuh"
#include "helpers.cuh"
#include "sh.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>

namespace cg = cooperative_groups;

__global__ void compute_cov2d_bounds_kernel(
    const unsigned num_pts, const float* __restrict__ covs2d, float* __restrict__ conics, float* __restrict__ radii
) {
    unsigned row = cg::this_grid().thread_rank();
    if (row >= num_pts) {
        return;
    }
    int index = row * 3;
    float3 conic;
    float radius;
    float3 cov2d{
        (float)covs2d[index], (float)covs2d[index + 1], (float)covs2d[index + 2]
    };
    compute_cov2d_bounds(cov2d, conic, radius);
    conics[index] = conic.x;
    conics[index + 1] = conic.y;
    conics[index + 2] = conic.z;
    radii[row] = radius;
}

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output radii
compute_cov2d_bounds_tensor(const int num_pts, torch::Tensor &covs2d) {
    CHECK_INPUT(covs2d);
    torch::Tensor conics = torch::zeros(
        {num_pts, covs2d.size(1)}, covs2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor radii =
        torch::zeros({num_pts, 1}, covs2d.options().dtype(torch::kFloat32));

    int blocks = (num_pts + N_THREADS - 1) / N_THREADS;

    compute_cov2d_bounds_kernel<<<blocks, N_THREADS>>>(
        num_pts,
        covs2d.contiguous().data_ptr<float>(),
        conics.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<float>()
    );
    return std::make_tuple(conics, radii);
}

torch::Tensor compute_sh_forward_tensor(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &coeffs
) {
    unsigned num_bases = num_sh_bases(degree);
    if (coeffs.ndimension() != 3 || coeffs.size(0) != num_points ||
        coeffs.size(1) != num_bases || coeffs.size(2) != 3) {
        AT_ERROR("coeffs must have dimensions (N, D, 3)");
    }
    torch::Tensor colors = torch::empty({num_points, 3}, coeffs.options());
    compute_sh_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        degree,
        degrees_to_use,
        (float3 *)viewdirs.contiguous().data_ptr<float>(),
        coeffs.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>()
    );
    return colors;
}

torch::Tensor compute_sh_backward_tensor(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &v_colors
) {
    if (viewdirs.ndimension() != 2 || viewdirs.size(0) != num_points ||
        viewdirs.size(1) != 3) {
        AT_ERROR("viewdirs must have dimensions (N, 3)");
    }
    if (v_colors.ndimension() != 2 || v_colors.size(0) != num_points ||
        v_colors.size(1) != 3) {
        AT_ERROR("v_colors must have dimensions (N, 3)");
    }
    unsigned num_bases = num_sh_bases(degree);
    torch::Tensor v_coeffs =
        torch::zeros({num_points, num_bases, 3}, v_colors.options());
    compute_sh_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        degree,
        degrees_to_use,
        (float3 *)viewdirs.contiguous().data_ptr<float>(),
        v_colors.contiguous().data_ptr<float>(),
        v_coeffs.contiguous().data_ptr<float>()
    );
    return v_coeffs;
}


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &scales,
    const float glob_scale,
    torch::Tensor &quats,
    torch::Tensor &viewmat,
    torch::Tensor &projmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_height,
    const unsigned img_width,
    const std::tuple<int, int, int> tile_bounds,
    const float clip_thresh
) {
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    float4 intrins = {fx, fy, cx, cy};

    // Triangular covariance.
    torch::Tensor cov3d_d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor depths_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));

    project_gaussians_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)means3d.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        projmat.contiguous().data_ptr<float>(),
        intrins,
        img_size_dim3,
        tile_bounds_dim3,
        clip_thresh,
        // Outputs.
        cov3d_d.contiguous().data_ptr<float>(),
        (float2 *)xys_d.contiguous().data_ptr<float>(),
        depths_d.contiguous().data_ptr<float>(),
        radii_d.contiguous().data_ptr<int>(),
        (float3 *)conics_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(
        cov3d_d, xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &scales,
    const float glob_scale,
    torch::Tensor &quats,
    torch::Tensor &viewmat,
    torch::Tensor &projmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_height,
    const unsigned img_width,
    torch::Tensor &cov3d,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_depth,
    torch::Tensor &v_conic
) {
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    float4 intrins = {fx, fy, cx, cy};

    // const auto num_cov3d = num_points * 6;

    // Triangular covariance.
    torch::Tensor v_cov2d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_cov3d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_mean3d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_scale =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_quat =
        torch::zeros({num_points, 4}, means3d.options().dtype(torch::kFloat32));

    project_gaussians_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)means3d.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        projmat.contiguous().data_ptr<float>(),
        intrins,
        img_size_dim3,
        cov3d.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        v_depth.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        // Outputs.
        (float3 *)v_cov2d.contiguous().data_ptr<float>(),
        v_cov3d.contiguous().data_ptr<float>(),
        (float3 *)v_mean3d.contiguous().data_ptr<float>(),
        (float3 *)v_scale.contiguous().data_ptr<float>(),
        (float4 *)v_quat.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat);
}

std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds
) {
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(cum_tiles_hit);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    torch::Tensor gaussian_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));

    map_gaussian_to_intersects<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float2 *)xys.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        cum_tiles_hit.contiguous().data_ptr<int32_t>(),
        tile_bounds_dim3,
        // Outputs.
        isect_ids_unsorted.contiguous().data_ptr<int64_t>(),
        gaussian_ids_unsorted.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(isect_ids_unsorted, gaussian_ids_unsorted);
}

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects, const torch::Tensor &isect_ids_sorted
) {
    CHECK_INPUT(isect_ids_sorted);
    torch::Tensor tile_bins = torch::zeros(
        {num_intersects, 2}, isect_ids_sorted.options().dtype(torch::kInt32)
    );
    get_tile_bin_edges<<<
        (num_intersects + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_intersects,
        isect_ids_sorted.contiguous().data_ptr<int64_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>()
    );
    return tile_bins;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    rasterize_forward<<<tile_bounds_dim3, block_dim3>>>(
        tile_bounds_dim3,
        img_size_dim3,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        (float3 *)out_img.contiguous().data_ptr<float>(),
        *(float3 *)background.contiguous().data_ptr<float>()
    );

    return std::make_tuple(out_img, final_Ts, final_idx);
}

/**
 * @brief (BATCH VERSION) C++ Host Function: Launches the batch rasterization CUDA kernel.
 * * This function orchestrates the rendering of a batch of images. It allocates memory
 * for the entire batch's output and launches a single, large CUDA kernel grid where
 * each thread block is responsible for rendering one tile from one image in the batch.
 * * @param batch_size The number of images in the batch.
 * @param tile_bounds Dimensions of the tile grid for a SINGLE image (tiles_x, tiles_y, 1).
 * @param block Dimensions of a CUDA thread block, corresponding to tile size in pixels.
 * @param img_size Dimensions of a SINGLE image in pixels (width, height, 1).
 * @param gaussian_ids_sorted A flattened, sorted list of global Gaussian IDs for all intersections in the batch.
 * @param tile_bins A flattened lookup table mapping each GLOBAL tile ID to a range in gaussian_ids_sorted.
 * @param xys Flattened 2D centers for all Gaussians in the batch.
 * @param conics Flattened conic parameters for all Gaussians.
 * @param colors Flattened color values for all Gaussians.
 * @param opacities Flattened opacity values for all Gaussians.
 * @param background A tensor of background colors, one for each image in the batch, shape (batch_size, channels).
 * @return A tuple of Tensors (out_img, final_Ts, final_idx), all with a batch dimension.
 */
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> rasterize_batch_forward_sum_tensor(
    const int batch_size,
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background
) {
    // --- 1. Input Validation and Parameter Setup ---
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;
    const int num_tiles_per_image = tile_bounds_dim3.x * tile_bounds_dim3.y;
    const int num_total_tiles = batch_size * num_tiles_per_image;

    // --- 2. Allocate Output Tensors for the Entire Batch ---
    // Note the added batch_size dimension for all outputs.
    torch::Tensor out_img = torch::zeros(
        {batch_size, img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {batch_size, img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {batch_size, img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    // --- 3. Launch the Batch-Aware CUDA Kernel ---
    // The grid dimension is now the total number of tiles across the entire batch.
    // Each block will process one tile from one image.
    rasterize_batch_forward_sum_kernel<<<num_total_tiles, block_dim3>>>(
        // Batch-specific parameters
        batch_size,
        num_tiles_per_image,
        // Original parameters
        tile_bounds_dim3,
        img_size_dim3,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        // Output pointers
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        (float3 *)out_img.contiguous().data_ptr<float>(),
        // Background pointer (now an array)
        (const float3 *)background.contiguous().data_ptr<float>()
    );

    return std::make_tuple(out_img, final_Ts, final_idx);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> rasterize_forward_sum_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    rasterize_forward_sum<<<tile_bounds_dim3, block_dim3>>>(
        tile_bounds_dim3,
        img_size_dim3,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        (float3 *)out_img.contiguous().data_ptr<float>(),
        *(float3 *)background.contiguous().data_ptr<float>()
    );

    return std::make_tuple(out_img, final_Ts, final_idx);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> nd_rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background
) {
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    nd_rasterize_forward<<<tile_bounds_dim3, block_dim3>>>(
        tile_bounds_dim3,
        img_size_dim3,
        channels,
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        out_img.contiguous().data_ptr<float>(),
        background.contiguous().data_ptr<float>()
    );

    return std::make_tuple(out_img, final_Ts, final_idx);
}


std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    nd_rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned BLOCK_H,
        const unsigned BLOCK_W,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_alpha // dL_dout_alpha
    ) {

    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + BLOCK_W - 1) / BLOCK_W,
        (img_height + BLOCK_H - 1) / BLOCK_H,
        1
    };
    const dim3 block(BLOCK_W, BLOCK_H, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());

    torch::Tensor workspace;
    if (channels > 3) {
        workspace = torch::zeros(
            {img_height, img_width, channels},
            xys.options().dtype(torch::kFloat32)
        );
    } else {
        workspace = torch::zeros({0}, xys.options().dtype(torch::kFloat32));
    }

    nd_rasterize_backward_kernel<<<tile_bounds, block>>>(
        tile_bounds,
        img_size,
        channels,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        v_output.contiguous().data_ptr<float>(),
        v_output_alpha.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>(),
        workspace.data_ptr<float>()
    );

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity);
}

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned BLOCK_H,
        const unsigned BLOCK_W,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_alpha // dL_dout_alpha
    ) {

    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2 || colors.size(1) != 3) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + BLOCK_W - 1) / BLOCK_W,
        (img_height + BLOCK_H - 1) / BLOCK_H,
        1
    };
    const dim3 block(BLOCK_W, BLOCK_H, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());

    rasterize_backward_kernel<<<tile_bounds, block>>>(
        tile_bounds,
        img_size,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        *(float3 *)background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        (float3 *)v_output.contiguous().data_ptr<float>(),
        v_output_alpha.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        (float3 *)v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity);
}

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    rasterize_backward_sum_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned BLOCK_H,
        const unsigned BLOCK_W,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_alpha
    ){

    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2 || colors.size(1) != 3) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + BLOCK_W - 1) / BLOCK_W,
        (img_height + BLOCK_H - 1) / BLOCK_H,
        1
    };
    const dim3 block(BLOCK_W, BLOCK_H, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());

    rasterize_backward_sum_kernel<<<tile_bounds, block>>>(
        tile_bounds,
        img_size,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        *(float3 *)background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        (float3 *)v_output.contiguous().data_ptr<float>(),
        v_output_alpha.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        (float3 *)v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity);
}

// 主机端Host代码，运行在CPU上，负责创建torch::Tensor，分配好存储输入数据和接收结果的内存空间
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_2d_forward_tensor(
    const int num_points,
    torch::Tensor &means2d,
    torch::Tensor &L_elements,
    const unsigned img_height,
    const unsigned img_width,
    const std::tuple<int, int, int> tile_bounds,
    const float clip_thresh
) {
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    // Triangular covariance.
    // torch::Tensor cov3d_d =
    //     torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor depths_d =
        torch::zeros({num_points}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({num_points}, means2d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, means2d.options().dtype(torch::kInt32));

    // 定义当个 kernel 需要完成的工作
    project_gaussians_2d_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float2 *)means2d.contiguous().data_ptr<float>(),
        (float3 *)L_elements.contiguous().data_ptr<float>(),
        img_size_dim3,
        tile_bounds_dim3,
        clip_thresh,
        // Outputs.
        (float2 *)xys_d.contiguous().data_ptr<float>(),
        depths_d.contiguous().data_ptr<float>(),
        radii_d.contiguous().data_ptr<int>(),
        (float3 *)conics_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(
        xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d
    );
}

// 主机端Host代码，Batch版本
// 运行在CPU上，负责为整个批次的数据创建torch::Tensor，并启动内核
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_2d_batch_forward_tensor(
    const int batch_size,                      // 新增：批次大小
    const int num_points_per_image,            // 修改：现在表示每张图的点数
    torch::Tensor &means2d,                    // 形状应为 (batch_size * num_points_per_image, 2)
    torch::Tensor &L_elements,                 // 形状应为 (batch_size * num_points_per_image, 3)
    const unsigned img_height,
    const unsigned img_width,
    const std::tuple<int, int, int> tile_bounds,
    const float clip_thresh
) {
    // 计算需要处理的总点数
    const int total_points = batch_size * num_points_per_image;

    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    // 为整个批次的结果分配内存
    // 注意：所有张量的大小都基于 total_points
    torch::Tensor xys_d =
        torch::zeros({total_points, 2}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor depths_d =
        torch::zeros({total_points}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({total_points}, means2d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({total_points, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({total_points}, means2d.options().dtype(torch::kInt32));

    // 启动内核，处理 total_points 个点
    project_gaussians_2d_forward_kernel<<<
        (total_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        total_points, // 传递总点数
        (float2 *)means2d.contiguous().data_ptr<float>(),
        (float3 *)L_elements.contiguous().data_ptr<float>(),
        img_size_dim3,
        tile_bounds_dim3,
        clip_thresh,
        // Outputs.
        (float2 *)xys_d.contiguous().data_ptr<float>(),
        depths_d.contiguous().data_ptr<float>(),
        radii_d.contiguous().data_ptr<int>(),
        (float3 *)conics_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(
        xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_2d_scale_rot_forward_tensor(
    const int num_points,
    torch::Tensor &means2d,
    torch::Tensor &scales2d,
    torch::Tensor &rotation,
    const unsigned img_height,
    const unsigned img_width,
    const std::tuple<int, int, int> tile_bounds,
    const float clip_thresh
) {
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    // Triangular covariance.
    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor depths_d =
        torch::zeros({num_points}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({num_points}, means2d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, means2d.options().dtype(torch::kInt32));

    project_gaussians_2d_scale_rot_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float2 *)means2d.contiguous().data_ptr<float>(),
        (float2 *)scales2d.contiguous().data_ptr<float>(),
        (float *)rotation.contiguous().data_ptr<float>(),
        img_size_dim3,
        tile_bounds_dim3,
        clip_thresh,
        // Outputs.
        (float2 *)xys_d.contiguous().data_ptr<float>(),
        depths_d.contiguous().data_ptr<float>(),
        radii_d.contiguous().data_ptr<int>(),
        (float3 *)conics_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(
        xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d
    );
}

// 主机端Host代码，Batch版本，适配 Scale + Rotation
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_2d_scale_rot_batch_forward_tensor(
    const int batch_size,               // 批次大小
    const int num_points_per_image,     // 每张图的点数
    torch::Tensor &means2d,             // [B * N, 2]
    torch::Tensor &scales2d,            // [B * N, 2]  <-- 替换了 L_elements
    torch::Tensor &rotation,            // [B * N, 1]  <-- 新增
    const unsigned img_height,
    const unsigned img_width,
    const std::tuple<int, int, int> tile_bounds,
    const float clip_thresh
) {
    // 1. 计算总点数 (Batch * N)
    const int total_points = batch_size * num_points_per_image;

    // 2. 准备各种维度参数
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    // 3. 为整个批次分配输出内存 (大小基于 total_points)
    torch::Tensor xys_d =
        torch::zeros({total_points, 2}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor depths_d =
        torch::zeros({total_points}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({total_points}, means2d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({total_points, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({total_points}, means2d.options().dtype(torch::kInt32));

    // 4. 启动 Scale-Rot 版本的 Kernel
    project_gaussians_2d_scale_rot_forward_kernel<<<
        (total_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        total_points, // 传入总点数
        (float2 *)means2d.contiguous().data_ptr<float>(),
        (float2 *)scales2d.contiguous().data_ptr<float>(), // 传入 scales
        (float *)rotation.contiguous().data_ptr<float>(),  // 传入 rotation
        img_size_dim3,
        tile_bounds_dim3,
        clip_thresh,
        // Outputs.
        (float2 *)xys_d.contiguous().data_ptr<float>(),
        depths_d.contiguous().data_ptr<float>(),
        radii_d.contiguous().data_ptr<int>(),
        (float3 *)conics_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(
        xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d
    );
}


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_2d_backward_tensor(
    const int num_points,
    torch::Tensor &means2d,
    torch::Tensor &L_elements,
    const unsigned img_height,
    const unsigned img_width,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_depth,
    torch::Tensor &v_conic
) {
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    // Triangular covariance.
    torch::Tensor v_cov2d =
        torch::zeros({num_points, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor v_L_elements =
        torch::zeros({num_points, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor v_mean2d =
        torch::zeros({num_points, 2}, means2d.options().dtype(torch::kFloat32));

    project_gaussians_2d_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float2 *)means2d.contiguous().data_ptr<float>(),
        (float3 *)L_elements.contiguous().data_ptr<float>(),
        img_size_dim3,
        radii.contiguous().data_ptr<int32_t>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        v_depth.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        // Outputs.
        (float3 *)v_cov2d.contiguous().data_ptr<float>(),
        (float2 *)v_mean2d.contiguous().data_ptr<float>(),
        (float3 *)v_L_elements.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_cov2d, v_mean2d, v_L_elements);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_2d_scale_rot_backward_tensor(
    const int num_points,
    torch::Tensor &means2d,
    torch::Tensor &scales2d,
    torch::Tensor &rotation,
    const unsigned img_height,
    const unsigned img_width,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_depth,
    torch::Tensor &v_conic
) {
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    // Triangular covariance.
    torch::Tensor v_cov2d =
        torch::zeros({num_points, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor v_scale =
        torch::zeros({num_points, 2}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor v_rot =
        torch::zeros({num_points, 1}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor v_mean2d =
        torch::zeros({num_points, 2}, means2d.options().dtype(torch::kFloat32));

    project_gaussians_2d_scale_rot_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float2 *)means2d.contiguous().data_ptr<float>(),
        (float2 *)scales2d.contiguous().data_ptr<float>(),
        (float *)rotation.contiguous().data_ptr<float>(),
        img_size_dim3,
        radii.contiguous().data_ptr<int32_t>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        v_depth.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        // Outputs.
        (float3 *)v_cov2d.contiguous().data_ptr<float>(),
        (float2 *)v_mean2d.contiguous().data_ptr<float>(),
        (float2 *)v_scale.contiguous().data_ptr<float>(),
        (float *)v_rot.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_cov2d, v_mean2d, v_scale, v_rot);
}

/**
 * @brief (BATCH VERSION) C++ Host Function: Launches the batch backward kernel for Scale-Rot 2D projection.
 * * @param batch_size The number of images in the batch.
 * @param num_points_per_image The number of Gaussians per image.
 * @param means2d The 2D means from the forward pass, shape (B * N, 2).
 * @param scales2d The Scales from the forward pass, shape (B * N, 2).
 * @param rotation The Rotation from the forward pass, shape (B * N, 1).
 * @param img_height Image height.
 * @param img_width Image width.
 * @param radii The radii from the forward pass, shape (B * N, 1).
 * @param conics The conics from the forward pass, shape (B * N, 3).
 * @param v_xy Incoming gradient w.r.t. the 2D xy positions.
 * @param v_depth Incoming gradient w.r.t. the depths.
 * @param v_conic Incoming gradient w.r.t. the conics.
 * @return A tuple of gradient Tensors (v_mean2d, v_scale, v_rot).
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
project_gaussians_2d_scale_rot_batch_backward_tensor(
    const int batch_size,
    const int num_points_per_image,
    const torch::Tensor &means2d,
    const torch::Tensor &scales2d,
    const torch::Tensor &rotation,
    const unsigned img_height,
    const unsigned img_width,
    const torch::Tensor &radii,
    const torch::Tensor &conics,
    const torch::Tensor &v_xy,
    const torch::Tensor &v_depth,
    const torch::Tensor &v_conic
) {
    // --- 1. Parameter Setup ---
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    // 计算 Batch 总点数
    const int total_points = batch_size * num_points_per_image;

    // --- 2. Allocate Output Gradient Tensors for the Entire Batch ---
    // 中间变量 v_cov2d (虽然不需要返回给 Python，但计算链式法则需要它)
    torch::Tensor v_cov2d =
        torch::zeros({total_points, 3}, means2d.options().dtype(torch::kFloat32));
    
    // 输出梯度：Scale (B*N, 2)
    torch::Tensor v_scale =
        torch::zeros({total_points, 2}, means2d.options().dtype(torch::kFloat32));
    
    // 输出梯度：Rotation (B*N, 1)
    torch::Tensor v_rot =
        torch::zeros({total_points, 1}, means2d.options().dtype(torch::kFloat32));
    
    // 输出梯度：Mean2D (B*N, 2)
    torch::Tensor v_mean2d =
        torch::zeros({total_points, 2}, means2d.options().dtype(torch::kFloat32));

    // --- 3. Launch the Batch-Aware Backward Kernel ---
    // 注意：使用 project_gaussians_2d_scale_rot_backward_kernel
    project_gaussians_2d_scale_rot_backward_kernel<<<
        (total_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        total_points,
        (float2 *)means2d.contiguous().data_ptr<float>(),
        (float2 *)scales2d.contiguous().data_ptr<float>(),
        (float *)rotation.contiguous().data_ptr<float>(),
        img_size_dim3,
        radii.contiguous().data_ptr<int32_t>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        v_depth.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        // Outputs (gradients)
        (float3 *)v_cov2d.contiguous().data_ptr<float>(),
        (float2 *)v_mean2d.contiguous().data_ptr<float>(),
        (float2 *)v_scale.contiguous().data_ptr<float>(),
        (float *)v_rot.contiguous().data_ptr<float>()
    );

    // 返回需要回传给 Python 优化器的梯度
    return std::make_tuple(v_mean2d, v_scale, v_rot);
}

/**
 * @brief (BATCH VERSION) C++ Host Function: Launches the kernel to map Gaussians to intersections for a batch.
 * * This function takes flattened tensors representing a batch of images and launches a kernel
 * to create a list of all Gaussian-tile intersections.
 * * @param batch_size The number of images in the batch.
 * @param num_points_per_image The number of Gaussians per image.
 * @param num_intersects The total number of intersections across the entire batch.
 * @param xys Flattened 2D centers for all Gaussians in the batch.
 * @param depths Flattened depth values for all Gaussians.
 * @param radii Flattened radii for all Gaussians.
 * @param cum_tiles_hit Cumulative sum of tiles hit by each Gaussian.
 * @param tile_bounds Dimensions of the tile grid for a SINGLE image.
 * @return A tuple of Tensors (isect_ids_unsorted, gaussian_ids_unsorted) for the entire batch.
 */
std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_batch_tensor(
    const int batch_size,
    const int num_points_per_image,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds
) {
    // Input validation
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(cum_tiles_hit);

    // Convert tile_bounds tuple to CUDA's dim3 struct
    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    // Allocate output tensors on the GPU
    torch::Tensor gaussian_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));

    const int total_points = batch_size * num_points_per_image;

    // Launch the batch-aware CUDA kernel
    map_gaussian_to_intersects_batch_kernel<<<
        (total_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        // Batch-specific parameters
        total_points,
        num_points_per_image,
        // Original parameters
        (float2 *)xys.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        cum_tiles_hit.contiguous().data_ptr<int32_t>(),
        tile_bounds_dim3,
        // Outputs
        isect_ids_unsorted.contiguous().data_ptr<int64_t>(),
        gaussian_ids_unsorted.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(isect_ids_unsorted, gaussian_ids_unsorted);
}


/**
 * @brief (BATCH VERSION) C++ Host Function: Launches the kernel to create tile bins for a batch.
 * * This function reuses the original `get_tile_bin_edges` as its logic is generic.
 * The key difference is that it allocates a much larger `tile_bins` tensor to accommodate
 * all tiles from all images in the batch.
 * * @param batch_size The number of images in the batch.
 * @param num_tiles_per_image The number of tiles in a single image.
 * @param num_intersects The total number of intersections across the entire batch.
 * @param isect_ids_sorted The sorted list of global intersection IDs.
 * @return A Tensor representing the tile bins for the entire batch.
 */
torch::Tensor get_tile_bin_edges_batch_tensor(
    int batch_size,
    int num_tiles_per_image,
    int num_intersects, 
    const torch::Tensor &isect_ids_sorted
) {
    CHECK_INPUT(isect_ids_sorted);

    // Allocate a larger tile_bins tensor for the entire batch
    const int num_total_tiles = batch_size * num_tiles_per_image;
    torch::Tensor tile_bins = torch::zeros(
        {num_total_tiles, 2}, isect_ids_sorted.options().dtype(torch::kInt32)
    );

    // REUSE the original kernel. Its logic is generic and works with global tile IDs.
    get_tile_bin_edges<<<
        (num_intersects + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_intersects,
        isect_ids_sorted.contiguous().data_ptr<int64_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>()
    );
    return tile_bins;
}


/**
 * @brief (BATCH VERSION) C++ Host Function: Launches the batch backward kernel for 2D projection.
 * * This function computes the gradients for the 2D projection operation for an entire batch.
 * It allocates memory for the output gradients and launches the batch-aware backward kernel.
 * * @param batch_size The number of images in the batch.
 * @param num_points_per_image The number of Gaussians per image.
 * @param means2d The 2D means from the forward pass, shape (B * N, 2).
 * @param L_elements The Cholesky factors from the forward pass, shape (B * N, 3).
 * @param radii The radii from the forward pass, shape (B * N, 1).
 * @param conics The conics from the forward pass, shape (B * N, 3).
 * @param v_xy Incoming gradient w.r.t. the 2D xy positions.
 * @param v_depth Incoming gradient w.r.t. the depths.
 * @param v_conic Incoming gradient w.r.t. the conics.
 * @return A tuple of gradient Tensors (v_cov2d, v_mean2d, v_L_elements).
 */
std::tuple<torch::Tensor, torch::Tensor> project_gaussians_2d_batch_backward_tensor(
    const int batch_size,
    const int num_points_per_image,
    const torch::Tensor &means2d,
    const torch::Tensor &L_elements,
    const unsigned img_height,
    const unsigned img_width,
    const torch::Tensor &radii,
    const torch::Tensor &conics,
    const torch::Tensor &v_xy,
    const torch::Tensor &v_depth,
    const torch::Tensor &v_conic
) {
    // --- 1. Parameter Setup ---
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    const int total_points = batch_size * num_points_per_image;

    // --- 2. Allocate Output Gradient Tensors for the Entire Batch ---
    // Note: The intermediate v_cov2d is not returned to Python but is needed for computation.
    torch::Tensor v_cov2d =
        torch::zeros({total_points, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor v_L_elements =
        torch::zeros({total_points, 3}, means2d.options().dtype(torch::kFloat32));
    torch::Tensor v_mean2d =
        torch::zeros({total_points, 2}, means2d.options().dtype(torch::kFloat32));

    // --- 3. Launch the Batch-Aware Backward Kernel ---
    project_gaussians_2d_backward_kernel<<<
        (total_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        total_points,
        (float2 *)means2d.contiguous().data_ptr<float>(),
        (float3 *)L_elements.contiguous().data_ptr<float>(),
        img_size_dim3,
        radii.contiguous().data_ptr<int32_t>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        v_depth.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        // Outputs (gradients)
        (float3 *)v_cov2d.contiguous().data_ptr<float>(),
        (float2 *)v_mean2d.contiguous().data_ptr<float>(),
        (float3 *)v_L_elements.contiguous().data_ptr<float>()
    );

    // Return the gradients for the inputs that require them.
    return std::make_tuple(v_mean2d, v_L_elements);
}


/**
 * @brief (BATCH VERSION) C++ Host Function: Launches the batch backward kernel for rasterization.
 *
 * This function orchestrates the gradient calculation for the rasterization step across an entire
 * batch of images. It allocates memory for the output gradients (dL/d_xys, dL/d_conics, etc.)
 * and launches a single, large CUDA kernel grid. Each thread block in the grid is responsible
 * for computing gradients for a single tile from a single image in the batch.
 *
 * @param batch_size The number of images in the batch.
 * @param img_height Height of a single image.
 * @param img_width Width of a single image.
 * @param BLOCK_H Height of a tile/block.
 * @param BLOCK_W Width of a tile/block.
 * @param gaussians_ids_sorted Flattened, sorted list of global Gaussian IDs from the forward pass.
 * @param tile_bins Flattened lookup table mapping each GLOBAL tile ID to a range in gaussians_ids_sorted.
 * @param xys Flattened 2D centers for all Gaussians in the batch.
 * @param conics Flattened conic parameters for all Gaussians.
 * @param colors Flattened color values for all Gaussians.
 * @param opacities Flattened opacity values for all Gaussians.
 * @param background A tensor of background colors, one for each image, shape (batch_size, channels).
 * @param final_Ts The final transmittance values from the forward pass, shape (batch_size, H, W).
 * @param final_idx The index of the last contributing Gaussian per pixel, shape (batch_size, H, W).
 * @param v_output Incoming gradient w.r.t. the output color image, shape (batch_size, H, W, C).
 * @param v_output_alpha Incoming gradient w.r.t. the output alpha channel.
 * @return A tuple of gradient Tensors (v_xy, v_conic, v_colors, v_opacity).
 */
std::tuple<
    torch::Tensor, // dL_dxy
    torch::Tensor, // dL_dconic
    torch::Tensor, // dL_dcolors
    torch::Tensor  // dL_dopacity
>
rasterize_batch_backward_sum_tensor(
    const int batch_size,
    const unsigned img_height,
    const unsigned img_width,
    const unsigned BLOCK_H,
    const unsigned BLOCK_W,
    const torch::Tensor &gaussians_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background,
    const torch::Tensor &final_Ts,
    const torch::Tensor &final_idx,
    const torch::Tensor &v_output,
    const torch::Tensor &v_output_alpha
){
    // --- 1. Input Validation and Parameter Setup ---
    CHECK_INPUT(xys);
    CHECK_INPUT(colors);
    // Add checks for other tensors as needed...

    const int total_points = xys.size(0);
    const int channels = colors.size(1);
    
    const dim3 tile_bounds = {
        (img_width + BLOCK_W - 1) / BLOCK_W,
        (img_height + BLOCK_H - 1) / BLOCK_H,
        1
    };
    const dim3 block_dim = {BLOCK_W, BLOCK_H, 1};
    const dim3 img_size = {img_width, img_height, 1};

    const int num_tiles_per_image = tile_bounds.x * tile_bounds.y;
    const int num_total_tiles = batch_size * num_tiles_per_image;

    // --- 2. Allocate Output Gradient Tensors for the Entire Batch ---
    torch::Tensor v_xy = torch::zeros({total_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({total_points, 3}, xys.options());
    torch::Tensor v_colors = torch::zeros({total_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({total_points, 1}, xys.options());

    // --- 3. Launch the Batch-Aware Backward Kernel ---
    rasterize_batch_backward_sum_kernel<<<num_total_tiles, block_dim>>>(
        // Batch-specific parameters
        batch_size,
        num_tiles_per_image,
        // Original parameters
        tile_bounds,
        img_size,
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        (const float3 *)background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        (const float3 *)v_output.contiguous().data_ptr<float>(),
        v_output_alpha.contiguous().data_ptr<float>(),
        // Output gradient pointers
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        (float3 *)v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity);
}