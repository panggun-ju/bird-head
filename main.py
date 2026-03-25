import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
from dataclasses import dataclass

def eye_aspect_ratio(eye_landmarks, all_landmarks, w, h):
    def pt(idx):
        return np.array([all_landmarks[idx].x * w, all_landmarks[idx].y * h])
    p1, p2, p3, p4, p5, p6 = [pt(eye_landmarks[i]) for i in range(6)]
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MAX_BODY_HEIGHT = 250
MESH_RESOLUTION = 14
ALPHA_THRESHOLD = 50

@dataclass
class Particle:
    x: float; y: float; old_x: float; old_y: float; fixed: bool = False

class Spring:
    def __init__(self, p1, p2, stiffness=0.99):
        self.p1 = p1; self.p2 = p2
        self.rest = math.hypot(p1.x - p2.x, p1.y - p2.y)
        self.stiffness = stiffness
    def solve(self):
        dx, dy = self.p1.x - self.p2.x, self.p1.y - self.p2.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6: return
        diff = (self.rest - dist) / dist
        ox, oy = dx * 0.5 * diff * self.stiffness, dy * 0.5 * diff * self.stiffness
        if not self.p1.fixed: self.p1.x += ox; self.p1.y += oy
        if not self.p2.fixed: self.p2.x -= ox; self.p2.y -= oy

class SoftBody:
    def __init__(self, vertices, triangles, start_x, start_y):
        self.vertices, self.triangles = vertices, triangles
        self.particles, self.local, self.springs, self.neighbors, self.rest_sign = [], [], [], set(), []
        cx, cy = sum(v[0] for v in vertices)/len(vertices), sum(v[1] for v in vertices)/len(vertices)
        for vx, vy in vertices:
            px, py = vx - cx + start_x, vy - cy + start_y
            self.particles.append(Particle(px, py, px, py))
            self.local.append((vx - cx, vy - cy))
        edges = set()
        for a, b, c in triangles:
            ax, ay, bx, by, cx3, cy3 = vertices[a][0], vertices[a][1], vertices[b][0], vertices[b][1], vertices[c][0], vertices[c][1]
            self.rest_sign.append(1.0 if (bx-ax)*(cy3-ay)-(by-ay)*(cx3-ax) >= 0.0 else -1.0)
            for i, j in ((a, b), (b, c), (c, a)):
                m, n = min(i, j), max(i, j)
                edges.add((m, n)); self.neighbors.add((m, n))
        for i, j in edges: self.springs.append(Spring(self.particles[i], self.particles[j], 0.99))

    def _prevent_triangle_flip(self):
        for tri_i, (a, b, c) in enumerate(self.triangles):
            pa, pb, pc = self.particles[a], self.particles[b], self.particles[c]
            if ((pb.x-pa.x)*(pc.y-pa.y)-(pb.y-pa.y)*(pc.x-pa.x)) * self.rest_sign[tri_i] >= 0.0: continue
            cx, cy = (pa.x+pb.x+pc.x)/3.0, (pa.y+pb.y+pc.y)/3.0
            for p in (pa, pb, pc):
                if not p.fixed: p.x += (cx-p.x)*0.6; p.y += (cy-p.y)*0.6

    def _repel_non_neighbors(self, min_dist=15.0, strength=0.5):
        min_d2 = min_dist**2; n = len(self.particles)
        for i in range(n):
            pi = self.particles[i]
            for j in range(i+1, n):
                if (i, j) in self.neighbors: continue
                dx, dy = pi.x - self.particles[j].x, pi.y - self.particles[j].y
                d2 = dx*dx + dy*dy
                if 0 < d2 < min_d2:
                    d = math.sqrt(d2); push = (min_dist - d) * strength; ux, uy = dx/d, dy/d
                    if not pi.fixed: pi.x += ux*push; pi.y += uy*push
                    if not self.particles[j].fixed: self.particles[j].x -= ux*push; self.particles[j].y -= uy*push

    def update(self, gravity=0.1, friction=0.999, shape_stiffness=0.92, iterations=25):
        for p in self.particles:
            if p.fixed: continue
            vx, vy = (p.x - p.old_x)*friction, (p.y - p.old_y)*friction
            p.old_x, p.old_y = p.x, p.y
            p.x += vx; p.y += vy + gravity
        cx, cy = sum(p.x for p in self.particles)/len(self.particles), sum(p.y for p in self.particles)/len(self.particles)
        sv, cv = 0.0, 0.0
        for p, (lx, ly) in zip(self.particles, self.local):
            dx, dy = p.x - cx, p.y - cy
            sv += lx*dy - ly*dx; cv += lx*dx + ly*dy
        sa, ca = math.sin(math.atan2(sv, cv)), math.cos(math.atan2(sv, cv))
        for p, (lx, ly) in zip(self.particles, self.local):
            if p.fixed: continue
            p.x += (cx + lx*ca - ly*sa - p.x) * shape_stiffness
            p.y += (cy + lx*sa + ly*ca - p.y) * shape_stiffness
        for _ in range(iterations):
            for sp in self.springs: sp.solve()
            self._prevent_triangle_flip(); self._repel_non_neighbors()

def resize_with_max_height(body, head, head_blink, max_h=MAX_BODY_HEIGHT):
    h, w = body.shape[:2]
    if h <= max_h: return body, head, head_blink, 1.0
    scale = max_h / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(body, (new_w, max_h), interpolation=cv2.INTER_AREA), cv2.resize(head, (max(1, int(head.shape[1]*scale)), max(1, int(head.shape[0]*scale))), interpolation=cv2.INTER_AREA), cv2.resize(head_blink, (max(1, int(head_blink.shape[1]*scale)), max(1, int(head_blink.shape[0]*scale))), interpolation=cv2.INTER_AREA), scale

def generate_dense_mesh_delaunay(image, resolution=MESH_RESOLUTION, alpha_th=ALPHA_THRESHOLD):
    h, w = image.shape[:2]; alpha = image[:,:,3] if image.shape[2]==4 else np.full((h,w), 255, dtype=np.uint8); mask = (alpha > alpha_th).astype(np.uint8)
    boundary = []
    for y in range(h):
        for x in range(w):
            if mask[y,x] and (x==0 or x==w-1 or y==0 or y==h-1 or not mask[y-1,x] or not mask[y+1,x] or not mask[y,x-1] or not mask[y,x+1]): boundary.append((x,y))
    pts = []
    for bx, by in boundary:
        if all((bx-px)**2 + (by-py)**2 >= (resolution*0.6)**2 for px, py in pts): pts.append((bx, by))
    for y in range(resolution//2, h, resolution):
        for x in range(resolution//2, w, resolution):
            if mask[y,x] and all((x-px)**2 + (y-py)**2 >= (resolution*0.7)**2 for px, py in pts): pts.append((x, y))
    if len(pts) < 4: return [(0,0),(w-1,0),(0,h-1),(w-1,h-1)], [(0,1,2),(1,3,2)]
    subdiv = cv2.Subdiv2D((0,0,w,h))
    for p in pts: subdiv.insert((float(p[0]), float(p[1])))
    triangles = []
    for t in subdiv.getTriangleList():
        cx, cy = int((t[0]+t[2]+t[4])/3), int((t[1]+t[3]+t[5])/3)
        if 0 <= cx < w and 0 <= cy < h and mask[cy,cx]:
            idx = []
            for i in range(0,6,2):
                best, bd = 0, float('inf')
                for j, (qx, qy) in enumerate(pts):
                    d = (t[i]-qx)**2 + (t[i+1]-qy)**2
                    if d < bd: bd, best = d, j
                idx.append(best)
            if idx[0]!=idx[1] and idx[1]!=idx[2] and idx[0]!=idx[2] and abs((pts[idx[1]][0]-pts[idx[0]][0])*(pts[idx[2]][1]-pts[idx[0]][1])-(pts[idx[2]][0]-pts[idx[0]][0])*(pts[idx[1]][1]-pts[idx[0]][1])) >= 1.0: triangles.append(tuple(idx))
    return pts, triangles

def draw_textured_mesh(frame, texture, particles, uvs, triangles):
    fh, fw = frame.shape[:2]
    for a, b, c in triangles:
        p1, p2, p3 = particles[a], particles[b], particles[c]; dst = np.float32([[p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y]])
        rect = cv2.boundingRect(dst)
        if rect[2]<=0 or rect[3]<=0 or rect[0]>=fw or rect[1]>=fh or rect[0]+rect[2]<=0 or rect[1]+rect[3]<=0: continue
        dst_l = dst - [rect[0], rect[1]]; src = np.float32([uvs[a], uvs[b], uvs[c]])
        try: mat = cv2.getAffineTransform(src, dst_l)
        except: continue
        warp = cv2.warpAffine(texture, mat, (rect[2], rect[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = np.zeros((rect[3], rect[2]), dtype=np.uint8); cv2.fillConvexPoly(mask, np.int32(dst_l), 255)
        x1, y1, x2, y2 = max(0, rect[0]), max(0, rect[1]), min(fw, rect[0]+rect[2]), min(fh, rect[1]+rect[3])
        ox1, oy1, ox2, oy2 = x1-rect[0], y1-rect[1], x1-rect[0]+(x2-x1), y1-rect[1]+(y2-y1)
        if ox2<=ox1 or oy2<=oy1: continue
        w_c, m_c, d_c = warp[oy1:oy2, ox1:ox2], mask[oy1:oy2, ox1:ox2], frame[y1:y2, x1:x2]
        alpha = (w_c[:,:,3]/255.0)*(m_c/255.0) if texture.shape[2]==4 else m_c/255.0
        for ch in range(3): d_c[:,:,ch] = alpha * w_c[:,:,ch] + (1.0-alpha) * d_c[:,:,ch]

def draw_body_contour(frame, particles, triangles, color=(0,0,0), thickness=3):
    ec = {}
    for a, b, c in triangles:
        for i, j in ((a,b), (b,c), (c,a)): m, n = min(i,j), max(i,j); ec[(m,n)] = ec.get((m,n), 0) + 1
    for (i, j), cnt in ec.items():
        if cnt == 1: cv2.line(frame, (int(particles[i].x), int(particles[i].y)), (int(particles[j].x), int(particles[j].y)), color, thickness, cv2.LINE_AA)

def overlay_transparent(background, overlay, x, y, angle=0.0):
    h, w = overlay.shape[:2]
    if angle != 0.0: overlay = cv2.warpAffine(overlay, cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    x1, y1, x2, y2 = max(0, int(x-w//2)), max(0, int(y-h//2)), min(background.shape[1], int(x-w//2)+w), min(background.shape[0], int(y-h//2)+h)
    ox1, oy1, ox2, oy2 = max(0, -(int(x-w//2))), max(0, -(int(y-h//2))), max(0, -(int(x-w//2)))+(x2-x1), max(0, -(int(y-h//2)))+(y2-y1)
    if x1>=x2 or y1>=y2: return background
    ol_c, bg_c = overlay[oy1:oy2, ox1:ox2], background[y1:y2, x1:x2]
    if overlay.shape[2]==4:
        a = ol_c[:,:,3]/255.0
        for c in range(3): bg_c[:,:,c] = a*ol_c[:,:,c] + (1-a)*bg_c[:,:,c]
    else: bg_c[:] = ol_c[:,:,:3]
    return background

def draw_alpha_contour(background, overlay, x, y, angle=0.0, color=(0,0,0), thickness=4):
    h, w = overlay.shape[:2]
    work = cv2.warpAffine(overlay, cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)) if angle!=0.0 else overlay
    if work.shape[2]<4: return background
    cnts, _ = cv2.findContours((work[:,:,3]>ALPHA_THRESHOLD).astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ox, oy = int(x-w//2), int(y-h//2)
    cv2.drawContours(background, [c + [ox, oy] for c in cnts], -1, color, thickness, cv2.LINE_AA)
    return background

def main():
    cap = cv2.VideoCapture(0)
    bird_body, bird_head, bird_head_blink = cv2.imread('assets/bird_body.png', -1), cv2.imread('assets/bird_head.png', -1), cv2.imread('assets/bird_head_blink.png', -1)
    bird_body, bird_head, bird_head_blink, _ = resize_with_max_height(bird_body, bird_head, bird_head_blink, MAX_BODY_HEIGHT)
    body_pts, body_tris = generate_dense_mesh_delaunay(bird_body); L = float(bird_body.shape[0])
    soft_body = SoftBody(body_pts, body_tris, 320, 100 + L*0.5)
    min_y, max_y = min(v[1] for v in body_pts), max(v[1] for v in body_pts)
    t_idx, b_idx = [i for i, p in enumerate(body_pts) if p[1]-min_y<=12], [i for i, p in enumerate(body_pts) if max_y-p[1]<=12]
    t_cx, b_cx = sum(body_pts[i][0] for i in t_idx)/len(t_idx), sum(body_pts[i][0] for i in b_idx)/len(b_idx)
    t_off, b_off = [body_pts[i][0]-t_cx for i in t_idx], [body_pts[i][0]-b_cx for i in b_idx]
    
    h_target = np.array([320.0, 100.0])
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'), running_mode=vision.RunningMode.VIDEO, num_hands=2))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='face_landmarker.task'), running_mode=vision.RunningMode.VIDEO, num_faces=1))
    blink_timer, frame_counter, is_grabbed = 0, 0, False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1); h, w = frame.shape[:2]; mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_counter += 1; h_res = hands.detect_for_video(mp_img, frame_counter*33); f_res = face.detect_for_video(mp_img, frame_counter*33)
        
        is_blinking = False
        if f_res.face_landmarks:
            lms = f_res.face_landmarks[0]
            if (eye_aspect_ratio(LEFT_EYE, lms, w, h) + eye_aspect_ratio(RIGHT_EYE, lms, w, h))/2.0 < 0.25: blink_timer = 5
        if blink_timer > 0: is_blinking, blink_timer = True, blink_timer-1
        
        h_det, h_pos, l_det, l_cnt, p_dist = False, np.array([w/2, h-100]), False, None, 0
        if h_res.hand_landmarks:
            for lms, hnd in zip(h_res.hand_landmarks, h_res.handedness):
                if hnd[0].category_name == 'Right': h_pos, h_det = np.array([lms[9].x*w, lms[9].y*h]), True
                else: tp, ip = np.array([lms[4].x*w, lms[4].y*h]), np.array([lms[8].x*w, lms[8].y*h]); p_dist, l_cnt, l_det = np.linalg.norm(tp-ip), (tp+ip)/2.0, True
        if not h_det: h_pos = h_target + [0, L]

        if l_det:
            if not is_grabbed:
                if np.linalg.norm(l_cnt-h_target) < 160 and 60 < p_dist < 240: is_grabbed = True
            elif p_dist > 280 or p_dist < 40: is_grabbed = False
        else: is_grabbed = False

        if is_grabbed: h_target += (l_cnt - h_target) * 0.5
        else:
            dist = np.linalg.norm(h_pos - h_target)
            if dist > L: h_target += ((h_pos-h_target)/dist) * min(12.5 * (1.0 + 1.4*(dist/L-1.0)), 48.0)
            if h_pos[1] < h_target[1] + L/2: h_target[1] -= 5.0
            
        dx, dy = h_pos[0]-h_target[0], h_pos[1]-h_target[1]; ang = math.atan2(dy, dx)-math.pi/2.0; ca, sa = math.cos(ang), math.sin(ang)

        # Double Physics & Double Rendering per camera frame
        for step in range(2):
            # Update physics (Sub-stepping internally)
            for i, off in zip(t_idx, t_off):
                p = soft_body.particles[i]; p.fixed, p.x, p.y, p.old_x, p.old_y = True, h_target[0]+off*ca, h_target[1]+off*sa, h_target[0]+off*ca, h_target[1]+off*sa
            for i, off in zip(b_idx, b_off):
                p = soft_body.particles[i]; p.fixed, p.x, p.y, p.old_x, p.old_y = True, h_pos[0]+off*ca, h_pos[1]+off*sa, h_pos[0]+off*ca, h_pos[1]+off*sa
            soft_body.update() # iterations=25 inside
            
            # Render bird on a fresh copy of the frame
            display_frame = frame.copy()
            draw_textured_mesh(display_frame, bird_body, soft_body.particles, body_pts, body_tris)
            draw_body_contour(display_frame, soft_body.particles, body_tris)
            h_img = bird_head_blink if is_blinking else bird_head
            h_ang = math.degrees(math.atan2(dy, dx))-90 if (abs(dx)+abs(dy))>1e-6 else 0.0
            overlay_transparent(display_frame, h_img, h_target[0], h_target[1], angle=h_ang*0.5)
            draw_alpha_contour(display_frame, h_img, h_target[0], h_target[1], angle=h_ang*0.5)

            cv2.imshow('Bird Head', display_frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            
    cap.release(); cv2.destroyAllWindows()

if __name__ == '__main__': main()
