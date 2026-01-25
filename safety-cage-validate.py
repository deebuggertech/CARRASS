import cv2
import numpy as np
import os

def compute_chamfer_distance(img_render_path, img_photo_path, img_bg_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    render = cv2.imread(img_render_path, cv2.IMREAD_UNCHANGED)
    photo = cv2.imread(img_photo_path)
    background = cv2.imread(img_bg_path)

    if any(img is None for img in [render, photo, background]):
        print("Error: Image loading failed. Check your paths.")
        return None

    h, w = render.shape[:2]
    photo = cv2.resize(photo, (w, h))
    background = cv2.resize(background, (w, h))

    render_alpha = render[:, :, 3] if render.shape[2] == 4 else cv2.cvtColor(render, cv2.COLOR_BGR2GRAY)
    r_edges_alpha = cv2.Canny(render_alpha, 50, 150)
    cv2.imwrite(f"{output_dir}/01_render_alpha_edges.png", r_edges_alpha)

    render_gray = cv2.cvtColor(render[:, :, :3], cv2.COLOR_BGR2GRAY)
    render_blur = cv2.bilateralFilter(render_gray, 7, 50, 50)
    r_edges_inner = cv2.Laplacian(render_blur, cv2.CV_8U, ksize=3)
    _, r_edges_inner = cv2.threshold(r_edges_inner, 30, 255, cv2.THRESH_BINARY)

    render_edges_combined = cv2.bitwise_or(r_edges_alpha, r_edges_inner)
    cv2.imwrite(f"{output_dir}/02_render_full_trace.png", render_edges_combined)

    diff = cv2.absdiff(photo, background)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(diff_gray, 20, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    cv2.imwrite(f"{output_dir}/03_photo_fg_mask.png", fg_mask)

    photo_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    photo_blur = cv2.GaussianBlur(photo_gray, (5, 5), 0)

    p_canny = cv2.Canny(photo_blur, 40, 120)
    cv2.imwrite(f"{output_dir}/04_photo_canny.png", p_canny)

    p_adapt = cv2.adaptiveThreshold(photo_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 3)
    cv2.imwrite(f"{output_dir}/05_photo_adaptive.png", p_adapt)

    gx = cv2.Sobel(photo_blur, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(photo_blur, cv2.CV_16S, 0, 1, ksize=3)
    p_sobel = cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)
    _, p_sobel = cv2.threshold(p_sobel, 40, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{output_dir}/06_photo_sobel.png", p_sobel)

    p_combined = cv2.bitwise_or(cv2.bitwise_or(p_canny, p_sobel), p_adapt)
    p_final_edges = cv2.bitwise_and(p_combined, fg_mask)

    nb, output, stats, _ = cv2.connectedComponentsWithStats(p_final_edges, connectivity=8)
    p_final_cleaned = np.zeros_like(p_final_edges)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= 15:
            p_final_cleaned[output == i] = 255
    cv2.imwrite(f"{output_dir}/07_photo_cleaned_final.png", p_final_cleaned)

    dist_trans = cv2.distanceTransform(cv2.bitwise_not(p_final_cleaned), cv2.DIST_L2, 5)
    dist_heatmap = cv2.applyColorMap(cv2.normalize(dist_trans, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{output_dir}/08_distance_heatmap.png", dist_heatmap)

    edge_coords = np.where(render_edges_combined > 0)
    if len(edge_coords[0]) == 0: return float('inf')

    dist_trans = cv2.distanceTransform(cv2.bitwise_not(p_final_cleaned), cv2.DIST_L2, 5)

    scale_factor = 0.5
    dist_down = cv2.resize(dist_trans, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    render_down = cv2.resize(render_edges_combined, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

    dist_heatmap = cv2.applyColorMap(cv2.normalize(dist_down, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{output_dir}/09_distance_heatmap_downsampled.png", dist_heatmap)

    edge_coords = np.where(render_down > 0)
    if len(edge_coords[0]) == 0:
        return float('inf')

    overlay = photo.copy()
    overlay[p_final_cleaned > 0] = [255, 255, 0]
    overlay[render_edges_combined > 0] = [0, 0, 255]
    cv2.imwrite(f"{output_dir}/10_final_overlay.png", overlay)

    max_dist_tolerance = 20
    truncated_dist = np.minimum(dist_down[edge_coords], max_dist_tolerance)

    chamfer_score = np.mean(truncated_dist)

    similarity = 100 * np.exp(-chamfer_score / 100)

    return similarity

if __name__ == "__main__":
    src_dir = "safety_cage_validate_src"
    out_dir = "safety_cage_validate_out"
    r, p, b = f"{src_dir}/render.png", f"{src_dir}/target.jpg", f"{src_dir}/background.jpg"
    score = compute_chamfer_distance(r, p, b, out_dir)
    if score is not None:
        print(f"Final Score: {score:.4f}")
