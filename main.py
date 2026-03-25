import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MAX_BODY_HEIGHT = 250
MESH_RESOLUTION = 18
ALPHA_THRESHOLD = 50
TARGET_FPS = 30.0
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
RETURN_RELEASE_RATIO = 0.90
RETURN_STOP_SPEED = 1.10
RETURN_ENTER_RATIO = 1.00
OVERALL_SCALE = 1.25
BODY_HEIGHT_RATIO_TO_CAMERA = 0.25


def eye_aspect_ratio(eye_landmarks, all_landmarks, w, h):
    def pt(idx):
        return np.array([all_landmarks[idx].x * w, all_landmarks[idx].y * h], dtype=np.float32)

    p1, p2, p3, p4, p5, p6 = [pt(eye_landmarks[i]) for i in range(6)]
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 1.0
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horizontal)


def get_blendshape_score(blendshape_list, name):
    for cat in blendshape_list:
        if cat.category_name == name:
            return float(cat.score)
    return 0.0


def resize_with_max_height(body, head, head_blink, max_h=MAX_BODY_HEIGHT):
    h, w = body.shape[:2]
    if h <= max_h:
        return body, head, head_blink, 1.0
    scale = max_h / float(h)
    new_w = max(1, int(w * scale))
    body_r = cv2.resize(body, (new_w, max_h), interpolation=cv2.INTER_AREA)
    head_r = cv2.resize(
        head,
        (max(1, int(head.shape[1] * scale)), max(1, int(head.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    blink_r = cv2.resize(
        head_blink,
        (max(1, int(head_blink.shape[1] * scale)), max(1, int(head_blink.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return body_r, head_r, blink_r, scale


def scale_image(img, scale):
    h, w = img.shape[:2]
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)


def generate_dense_mesh_delaunay(image, resolution=MESH_RESOLUTION, alpha_th=ALPHA_THRESHOLD):
    h, w = image.shape[:2]
    alpha = image[:, :, 3] if image.shape[2] == 4 else np.full((h, w), 255, dtype=np.uint8)
    mask = (alpha > alpha_th).astype(np.uint8)

    boundary = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                continue
            if (
                x == 0
                or x == w - 1
                or y == 0
                or y == h - 1
                or mask[y - 1, x] == 0
                or mask[y + 1, x] == 0
                or mask[y, x - 1] == 0
                or mask[y, x + 1] == 0
            ):
                boundary.append((x, y))

    points = []
    boundary_min_d2 = (resolution * 0.6) ** 2
    for bx, by in boundary:
        if all((bx - px) ** 2 + (by - py) ** 2 >= boundary_min_d2 for px, py in points):
            points.append((bx, by))

    interior_min_d2 = (resolution * 0.7) ** 2
    for y in range(resolution // 2, h, resolution):
        for x in range(resolution // 2, w, resolution):
            if mask[y, x] != 0 and all((x - px) ** 2 + (y - py) ** 2 >= interior_min_d2 for px, py in points):
                points.append((x, y))

    if len(points) < 4:
        return [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)], [(0, 1, 2), (1, 3, 2)]

    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    def nearest_idx(px, py):
        best_i = 0
        best_d = float("inf")
        for i, (qx, qy) in enumerate(points):
            d = (px - qx) ** 2 + (py - qy) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    triangles = []
    for t in subdiv.getTriangleList():
        cx = int((t[0] + t[2] + t[4]) / 3)
        cy = int((t[1] + t[3] + t[5]) / 3)
        if 0 <= cx < w and 0 <= cy < h and mask[cy, cx] != 0:
            i1 = nearest_idx(t[0], t[1])
            i2 = nearest_idx(t[2], t[3])
            i3 = nearest_idx(t[4], t[5])
            if i1 != i2 and i2 != i3 and i1 != i3:
                area2 = abs(
                    (points[i2][0] - points[i1][0]) * (points[i3][1] - points[i1][1])
                    - (points[i3][0] - points[i1][0]) * (points[i2][1] - points[i1][1])
                )
                if area2 >= 1.0:
                    triangles.append((i1, i2, i3))

    if not triangles:
        return [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)], [(0, 1, 2), (1, 3, 2)]
    return points, triangles


class ARAPDeformer:
    def __init__(self, vertices, triangles, start_x, start_y):
        self.triangles = triangles
        self.rest = np.array(vertices, dtype=np.float32)
        self.current = self.rest.copy()

        self.rest_center = np.mean(self.rest, axis=0)
        self.rest_local = self.rest - self.rest_center

        self.current_center = np.array([start_x, start_y], dtype=np.float32)
        self.current = self.rest_local + self.current_center

        self.n = self.rest.shape[0]
        self.neighbors = [[] for _ in range(self.n)]
        edge_set = set()
        for a, b, c in self.triangles:
            for i, j in ((a, b), (b, c), (c, a)):
                m, n2 = (i, j) if i < j else (j, i)
                edge_set.add((m, n2))
        for i, j in edge_set:
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)

        # Precompute Laplacian matrix (constant over time).
        self.lap_base = np.zeros((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            deg = len(self.neighbors[i])
            self.lap_base[i, i] = float(deg)
            for j in self.neighbors[i]:
                self.lap_base[i, j] -= 1.0

        # Cache structures for fast constrained solve (topology is fixed).
        self._cached_key = None
        self._free_idx = None
        self._cons_idx = None
        self._L_ff_inv = None

    def set_anchors(self, top_idx, top_pos, bot_idx, bot_pos):
        self.top_idx = top_idx
        self.top_pos = top_pos
        self.bot_idx = bot_idx
        self.bot_pos = bot_pos

    def _ensure_factorization(self, constraints):
        cons_sorted = sorted(constraints.keys())
        key = tuple(cons_sorted)
        if self._cached_key == key:
            return

        cons_idx = np.array(cons_sorted, dtype=np.int32)
        free_idx = np.array([i for i in range(self.n) if i not in constraints], dtype=np.int32)
        if free_idx.size == 0:
            self._cached_key = key
            self._cons_idx = cons_idx
            self._free_idx = free_idx
            self._L_ff_inv = None
            return

        L_ff = self.lap_base[np.ix_(free_idx, free_idx)]
        self._L_ff_inv = np.linalg.inv(L_ff)
        self._cached_key = key
        self._cons_idx = cons_idx
        self._free_idx = free_idx

    def solve(self, iterations=3):
        # Local-global ARAP (2D, uniform weights):
        # minimizes ||(x_i-x_j) - R_i(rest_i-rest_j)|| under anchor constraints.
        constraints = {}
        for k, idx in enumerate(self.top_idx):
            constraints[int(idx)] = self.top_pos[k]
        for k, idx in enumerate(self.bot_idx):
            constraints[int(idx)] = self.bot_pos[k]

        self._ensure_factorization(constraints)

        # Start from previous state + hard anchors.
        for idx, pos in constraints.items():
            self.current[idx] = pos

        rotations = [np.eye(2, dtype=np.float32) for _ in range(self.n)]

        for _ in range(iterations):
            # Local step: estimate per-vertex rotation by SVD.
            for i in range(self.n):
                nbrs = self.neighbors[i]
                if not nbrs:
                    rotations[i] = np.eye(2, dtype=np.float32)
                    continue
                S = np.zeros((2, 2), dtype=np.float32)
                xi = self.current[i]
                ri = self.rest[i]
                for j in nbrs:
                    p = (ri - self.rest[j]).reshape(2, 1)
                    q = (xi - self.current[j]).reshape(2, 1)
                    S += q @ p.T
                U, _, Vt = np.linalg.svd(S)
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    U[:, 1] *= -1.0
                    R = U @ Vt
                rotations[i] = R.astype(np.float32)

            # Global step: direct linear solve Lx=b with hard constraints.
            b = np.zeros((self.n, 2), dtype=np.float64)
            for i in range(self.n):
                for j in self.neighbors[i]:
                    p_ij = self.rest[i] - self.rest[j]
                    b[i] += 0.5 * ((rotations[i] + rotations[j]) @ p_ij)

            # Fast constrained solve using cached free/constrained partition.
            x_out = np.zeros(self.n, dtype=np.float64)
            y_out = np.zeros(self.n, dtype=np.float64)

            cons_vals_x = np.array([float(constraints[int(i)][0]) for i in self._cons_idx], dtype=np.float64)
            cons_vals_y = np.array([float(constraints[int(i)][1]) for i in self._cons_idx], dtype=np.float64)

            x_out[self._cons_idx] = cons_vals_x
            y_out[self._cons_idx] = cons_vals_y

            if self._free_idx.size > 0:
                L_fc = self.lap_base[np.ix_(self._free_idx, self._cons_idx)]
                b_free_x = b[self._free_idx, 0] - L_fc @ cons_vals_x
                b_free_y = b[self._free_idx, 1] - L_fc @ cons_vals_y
                x_out[self._free_idx] = self._L_ff_inv @ b_free_x
                y_out[self._free_idx] = self._L_ff_inv @ b_free_y

            self.current[:, 0] = x_out.astype(np.float32)
            self.current[:, 1] = y_out.astype(np.float32)


def draw_textured_mesh(frame, texture, vertices, uvs, triangles):
    fh, fw = frame.shape[:2]
    for a, b, c in triangles:
        p1 = vertices[a]
        p2 = vertices[b]
        p3 = vertices[c]
        dst = np.float32([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]]])
        rect = cv2.boundingRect(dst)
        x, y, w, h = rect
        if w <= 0 or h <= 0 or x >= fw or y >= fh or x + w <= 0 or y + h <= 0:
            continue
        dst_local = dst - np.array([x, y], dtype=np.float32)
        src = np.float32([uvs[a], uvs[b], uvs[c]])
        try:
            mat = cv2.getAffineTransform(src, dst_local)
        except cv2.error:
            continue
        warped = cv2.warpAffine(texture, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        tri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, np.int32(dst_local), 255)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)
        ox1 = x1 - x
        oy1 = y1 - y
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)
        if ox2 <= ox1 or oy2 <= oy1:
            continue

        warp_c = warped[oy1:oy2, ox1:ox2]
        mask_c = tri_mask[oy1:oy2, ox1:ox2]
        dst_c = frame[y1:y2, x1:x2]
        alpha = (warp_c[:, :, 3] / 255.0) * (mask_c / 255.0) if texture.shape[2] == 4 else (mask_c / 255.0)
        for ch in range(3):
            dst_c[:, :, ch] = alpha * warp_c[:, :, ch] + (1.0 - alpha) * dst_c[:, :, ch]


def draw_body_contour(frame, vertices, triangles, color=(0, 0, 0), thickness=3):
    edge_count = {}
    for a, b, c in triangles:
        for i, j in ((a, b), (b, c), (c, a)):
            m, n = min(i, j), max(i, j)
            edge_count[(m, n)] = edge_count.get((m, n), 0) + 1
    for (i, j), cnt in edge_count.items():
        if cnt == 1:
            p1 = vertices[i]
            p2 = vertices[j]
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness, cv2.LINE_AA)


def overlay_transparent(background, overlay, x, y, angle=0.0):
    ol_h, ol_w = overlay.shape[:2]
    if angle != 0.0:
        mat = cv2.getRotationMatrix2D((ol_w // 2, ol_h // 2), angle, 1.0)
        overlay = cv2.warpAffine(
            overlay,
            mat,
            (ol_w, ol_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
    x1 = max(0, int(x - ol_w // 2))
    y1 = max(0, int(y - ol_h // 2))
    x2 = min(background.shape[1], int(x - ol_w // 2) + ol_w)
    y2 = min(background.shape[0], int(y - ol_h // 2) + ol_h)
    ox1 = max(0, -(int(x - ol_w // 2)))
    oy1 = max(0, -(int(y - ol_h // 2)))
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)
    if x1 >= x2 or y1 >= y2:
        return background

    ol_crop = overlay[oy1:oy2, ox1:ox2]
    bg_crop = background[y1:y2, x1:x2]
    if overlay.shape[2] == 4:
        alpha = ol_crop[:, :, 3] / 255.0
        for c in range(3):
            bg_crop[:, :, c] = alpha * ol_crop[:, :, c] + (1.0 - alpha) * bg_crop[:, :, c]
    else:
        bg_crop[:] = ol_crop[:, :, :3]
    return background


def draw_alpha_contour(background, overlay, x, y, angle=0.0, color=(0, 0, 0), thickness=4):
    ol_h, ol_w = overlay.shape[:2]
    if angle != 0.0:
        mat = cv2.getRotationMatrix2D((ol_w // 2, ol_h // 2), angle, 1.0)
        work = cv2.warpAffine(
            overlay,
            mat,
            (ol_w, ol_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
    else:
        work = overlay
    if work.shape[2] < 4:
        return background
    binary = (work[:, :, 3] > ALPHA_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return background
    ox = int(x - ol_w // 2)
    oy = int(y - ol_h // 2)
    shifted = [cnt + [ox, oy] for cnt in contours]
    cv2.drawContours(background, shifted, -1, color, thickness, cv2.LINE_AA)
    return background


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    adaptive_body_h = max(64, int(cam_h * BODY_HEIGHT_RATIO_TO_CAMERA)) if cam_h > 0 else MAX_BODY_HEIGHT

    bird_body = cv2.imread("assets/bird_body.png", cv2.IMREAD_UNCHANGED)
    bird_head = cv2.imread("assets/bird_head.png", cv2.IMREAD_UNCHANGED)
    bird_head_blink = cv2.imread("assets/bird_head_blink.png", cv2.IMREAD_UNCHANGED)
    bird_head_grab = cv2.imread("assets/bird_head_grap.png", cv2.IMREAD_UNCHANGED)

    if bird_body is None or bird_head is None or bird_head_blink is None or bird_head_grab is None:
        raise FileNotFoundError("assets 이미지 파일을 찾을 수 없습니다.")

    bird_body, bird_head, bird_head_blink, scale = resize_with_max_height(
        bird_body, bird_head, bird_head_blink, adaptive_body_h
    )
    bird_head_grab = cv2.resize(
        bird_head_grab,
        (max(1, int(bird_head_grab.shape[1] * scale)), max(1, int(bird_head_grab.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )

    # Global size upscaling (body + all head sprites)
    bird_body = scale_image(bird_body, OVERALL_SCALE)
    bird_head = scale_image(bird_head, OVERALL_SCALE)
    bird_head_blink = scale_image(bird_head_blink, OVERALL_SCALE)
    bird_head_grab = scale_image(bird_head_grab, OVERALL_SCALE)

    body_points, body_tris = generate_dense_mesh_delaunay(bird_body, MESH_RESOLUTION, ALPHA_THRESHOLD)
    L = float(bird_body.shape[0])

    deformer = ARAPDeformer(body_points, body_tris, start_x=320, start_y=100 + L * 0.5)

    min_y = min(v[1] for v in body_points)
    max_y = max(v[1] for v in body_points)
    top_idx = [i for i, p in enumerate(body_points) if p[1] - min_y <= 12]
    bot_idx = [i for i, p in enumerate(body_points) if max_y - p[1] <= 12]
    if not top_idx:
        top_idx = [0]
    if not bot_idx:
        bot_idx = [len(body_points) - 1]

    top_cx = sum(body_points[i][0] for i in top_idx) / len(top_idx)
    bot_cx = sum(body_points[i][0] for i in bot_idx) / len(bot_idx)
    top_off = [body_points[i][0] - top_cx for i in top_idx]
    bot_off = [body_points[i][0] - bot_cx for i in bot_idx]

    # Head state: inertial spring-damper style return.
    head_pos = np.array([320.0, 100.0], dtype=np.float32)
    head_vel = np.zeros(2, dtype=np.float32)

    base_options_hands = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options_hands = vision.HandLandmarkerOptions(
        base_options=base_options_hands,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands = vision.HandLandmarker.create_from_options(options_hands)

    base_options_face = python.BaseOptions(model_asset_path="face_landmarker.task")
    options_face = vision.FaceLandmarkerOptions(
        base_options=base_options_face,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_mesh = vision.FaceLandmarker.create_from_options(options_face)

    blink_timer = 0
    frame_counter = 0
    is_head_grabbed = False
    is_returning = False

    while cap.isOpened():
        frame_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_counter += 1
        timestamp_ms = frame_counter * 33
        hand_results = hands.detect_for_video(mp_img, timestamp_ms)
        face_results = face_mesh.detect_for_video(mp_img, timestamp_ms)

        is_blinking = False
        if face_results.face_blendshapes and len(face_results.face_blendshapes) > 0:
            face_bs = face_results.face_blendshapes[0]
            left_blink = get_blendshape_score(face_bs, "eyeBlinkLeft")
            right_blink = get_blendshape_score(face_bs, "eyeBlinkRight")
            if left_blink >= 0.4 and right_blink >= 0.4:
                blink_timer = 4
        if blink_timer > 0:
            is_blinking = True
            blink_timer -= 1

        hand_detected = False
        hand_pos = np.array([w * 0.5, h - 100.0], dtype=np.float32)
        left_detected = False
        left_center = None
        pinch_dist = 0.0

        if hand_results.hand_landmarks and hand_results.handedness:
            for hand_lms, handedness in zip(hand_results.hand_landmarks, hand_results.handedness):
                label = handedness[0].category_name
                if label == "Right":
                    palm = hand_lms[9]
                    hand_pos = np.array([palm.x * w, palm.y * h], dtype=np.float32)
                    hand_detected = True
                elif label == "Left":
                    thumb = np.array([hand_lms[4].x * w, hand_lms[4].y * h], dtype=np.float32)
                    index = np.array([hand_lms[8].x * w, hand_lms[8].y * h], dtype=np.float32)
                    pinch_dist = float(np.linalg.norm(thumb - index))
                    left_center = (thumb + index) * 0.5
                    left_detected = True

        if not hand_detected:
            hand_pos = head_pos + np.array([0.0, L], dtype=np.float32)

        head_h, head_w = bird_head.shape[:2]
        head_radius = min(head_w, head_h) * 0.5
        grab_dist_th = head_radius * (2.0 / 3.0)

        if left_detected:
            inside_head = np.linalg.norm(left_center - head_pos) <= head_radius
            if not is_head_grabbed:
                if pinch_dist <= grab_dist_th and inside_head:
                    is_head_grabbed = True
            else:
                # 유지 조건은 핀치 거리만 사용 (head inside 조건 제거)
                if pinch_dist > (grab_dist_th * 1.25):
                    is_head_grabbed = False
        else:
            is_head_grabbed = False

        # Heuristic head control:
        # - grabbed: apply direct dx/dy velocity impulse
        # - released: return to rope-length target in ~1s with damped oscillation
        if is_head_grabbed and left_center is not None:
            is_returning = False
            drag = left_center - head_pos
            head_vel += drag * 0.30
            head_vel *= 0.80
        else:
            vec = hand_pos - head_pos
            dist = float(np.linalg.norm(vec))
            speed = float(np.linalg.norm(head_vel))

            # Return state latch:
            # - start when stretched beyond boundary
            # - keep returning until both near boundary and low speed
            if not is_returning and dist > L * RETURN_ENTER_RATIO:
                is_returning = True
            if is_returning and (dist <= L * RETURN_RELEASE_RATIO) and (speed <= RETURN_STOP_SPEED):
                is_returning = False

            if is_returning and dist > 1e-6:
                target = hand_pos - (vec / dist) * L
                # 조건 초과 시에만 움직임 + 아주 강한 damping.
                w0 = 6.2
                zeta = 1.35
                acc = (target - head_pos) * (w0 * w0) - 2.0 * zeta * w0 * head_vel
                head_vel += acc * (1.0 / 30.0)
                head_vel *= 0.70
            else:
                # 조건 미충족이면 완전 고정(속도 제거), damping 계산 없음.
                head_vel[:] = 0.0

        head_pos += head_vel

        # Head should not go below current body mesh center of mass.
        body_com_y = float(np.mean(deformer.current[:, 1]))
        if head_pos[1] > body_com_y:
            head_pos[1] = body_com_y
            if head_vel[1] > 0.0:
                head_vel[1] = 0.0

        dx = hand_pos[0] - head_pos[0]
        dy = hand_pos[1] - head_pos[1]
        body_angle = math.atan2(float(dy), float(dx)) - math.pi * 0.5
        ca = math.cos(body_angle)
        sa = math.sin(body_angle)

        top_pos = np.array([[head_pos[0] + off * ca, head_pos[1] + off * sa] for off in top_off], dtype=np.float32)
        bot_pos = np.array([[hand_pos[0] + off * ca, hand_pos[1] + off * sa] for off in bot_off], dtype=np.float32)

        deformer.set_anchors(top_idx, top_pos, bot_idx, bot_pos)
        deformer.solve(iterations=3)

        display = frame.copy()
        draw_textured_mesh(display, bird_body, deformer.current, body_points, body_tris)
        draw_body_contour(display, deformer.current, body_tris, color=(0, 0, 0), thickness=3)

        if is_head_grabbed:
            head_img = bird_head_grab
        else:
            head_img = bird_head_blink if is_blinking else bird_head
        head_angle = math.degrees(math.atan2(float(dy), float(dx))) - 90.0 if abs(dx) + abs(dy) > 1e-6 else 0.0
        overlay_transparent(display, head_img, head_pos[0], head_pos[1], angle=head_angle * 0.5)
        draw_alpha_contour(display, head_img, head_pos[0], head_pos[1], angle=head_angle * 0.5, color=(0, 0, 0), thickness=4)

        cv2.imshow("Bird Head", display)
        elapsed = time.perf_counter() - frame_start
        if elapsed < TARGET_FRAME_TIME:
            wait_ms = max(1, int((TARGET_FRAME_TIME - elapsed) * 1000.0))
        else:
            wait_ms = 1

        if cv2.waitKey(wait_ms) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
