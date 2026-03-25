import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time

def eye_aspect_ratio(eye_landmarks, all_landmarks, w, h):
    def pt(idx):
        return np.array([all_landmarks[idx].x * w, all_landmarks[idx].y * h])
    
    p1 = pt(eye_landmarks[0])
    p2 = pt(eye_landmarks[1])
    p3 = pt(eye_landmarks[2])
    p4 = pt(eye_landmarks[3])
    p5 = pt(eye_landmarks[4])
    p6 = pt(eye_landmarks[5])
    
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def overlay_transparent(background, overlay, x, y, pivot_x=None, pivot_y=None, angle=0.0):
    bg_h, bg_w, bg_channels = background.shape
    ol_h, ol_w, ol_channels = overlay.shape

    if pivot_x is None: pivot_x = ol_w // 2
    if pivot_y is None: pivot_y = ol_h // 2

    if angle != 0.0:
        M = cv2.getRotationMatrix2D((pivot_x, pivot_y), angle, 1.0)
        overlay = cv2.warpAffine(overlay, M, (ol_w, ol_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    x = int(x - pivot_x)
    y = int(y - pivot_y)

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + ol_w), min(bg_h, y + ol_h)
    
    ox1, oy1 = max(0, -x), max(0, -y)
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    if x1 >= bg_w or y1 >= bg_h or x2 <= 0 or y2 <= 0:
        return background

    ol_crop = overlay[oy1:oy2, ox1:ox2]
    bg_crop = background[y1:y2, x1:x2]

    if ol_channels == 4:
        alpha = ol_crop[:, :, 3] / 255.0
        for c in range(0, 3):
            bg_crop[:, :, c] = (alpha * ol_crop[:, :, c] + (1 - alpha) * bg_crop[:, :, c])
    else:
        background[y1:y2, x1:x2] = ol_crop[:, :, :3]

    return background

def main():
    cap = cv2.VideoCapture(0)
    
    bird_body = cv2.imread('assets/bird_body.png', cv2.IMREAD_UNCHANGED)
    bird_head = cv2.imread('assets/bird_head.png', cv2.IMREAD_UNCHANGED)
    bird_head_blink = cv2.imread('assets/bird_head_blink.png', cv2.IMREAD_UNCHANGED)
    
    N = 8 
    L = 300.0 
    segment_length = L / (N - 1)
    
    nodes = [np.array([320.0, 100.0 + i * segment_length]) for i in range(N)]
    velocities = [np.array([0.0, 0.0]) for _ in range(N)]
    
    gravity = np.array([0.0, 0.5])
    dt = 1.0
    
    head_pos_target = np.array([320.0, 100.0])
    
    base_options_hands = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options_hands = vision.HandLandmarkerOptions(
        base_options=base_options_hands,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    hands = vision.HandLandmarker.create_from_options(options_hands)
    
    base_options_face = python.BaseOptions(model_asset_path='face_landmarker.task')
    options_face = vision.FaceLandmarkerOptions(
        base_options=base_options_face,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    face_mesh = vision.FaceLandmarker.create_from_options(options_face)
    
    blink_timer = 0
    blink_threshold = 0.25
    is_head_grabbed = False
    
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_counter += 1
        timestamp_ms = frame_counter * 33

        hand_results = hands.detect_for_video(mp_image, timestamp_ms)
        face_results = face_mesh.detect_for_video(mp_image, timestamp_ms)
        
        is_blinking = False
        if face_results.face_landmarks and len(face_results.face_landmarks) > 0:
            landmarks = face_results.face_landmarks[0]
            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks, w, h)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            if avg_ear < blink_threshold:
                blink_timer = 5
            
        if blink_timer > 0:
            is_blinking = True
            blink_timer -= 1
            
        hand_detected = False
        hand_pos = np.array([w/2, h-100])
        
        left_hand_detected = False
        left_hand_center = None
        pinch_dist = 0
        
        if hand_results.hand_landmarks and hand_results.handedness:
            for hand_landmarks, handedness in zip(hand_results.hand_landmarks, hand_results.handedness):
                label = handedness[0].category_name
                if label == 'Right':
                    palm = hand_landmarks[9]
                    hand_pos = np.array([palm.x * w, palm.y * h])
                    hand_detected = True
                elif label == 'Left':
                    thumb = hand_landmarks[4]
                    index = hand_landmarks[8]
                    thumb_pos = np.array([thumb.x * w, thumb.y * h])
                    index_pos = np.array([index.x * w, index.y * h])
                    pinch_dist = np.linalg.norm(thumb_pos - index_pos)
                    left_hand_center = (thumb_pos + index_pos) / 2.0
                    left_hand_detected = True
        
        if not hand_detected:
            hand_pos = head_pos_target + np.array([0, L])

        head_radius = 80
        if left_hand_detected:
            dist_to_head = np.linalg.norm(left_hand_center - head_pos_target)
            
            if not is_head_grabbed:
                if dist_to_head < head_radius * 2.0 and 60 < pinch_dist < 240:
                    is_head_grabbed = True
            else:
                if pinch_dist > 280 or pinch_dist < 40:
                    is_head_grabbed = False
        else:
            is_head_grabbed = False

        if is_head_grabbed:
            head_pos_target += (left_hand_center - head_pos_target) * 0.5
        else:
            current_head = nodes[0]
            dist_to_hand = np.linalg.norm(hand_pos - current_head)
            
            if dist_to_hand > 2.0 * L:
                dir_to_hand = (hand_pos - current_head) / dist_to_hand
                head_pos_target += dir_to_hand * 5.0
                
            if hand_pos[1] < current_head[1] + L / 2:
                head_pos_target[1] -= 5.0
            
        nodes[0] = head_pos_target.copy()
        nodes[-1] = hand_pos.copy()
        
        for _ in range(5): 
            for i in range(1, N-1):
                nodes[i] += gravity * dt
                
            for i in range(N-1):
                diff = nodes[i+1] - nodes[i]
                dist = np.linalg.norm(diff)
                if dist > 0:
                    correction = (dist - segment_length) * (diff / dist) * 0.5
                    if i != 0: nodes[i] += correction
                    if i+1 != N-1: nodes[i+1] -= correction
                    
        body_h, body_w, _ = bird_body.shape
        strip_h = body_h // (N - 1)
        
        left_points = []
        right_points = []
        
        for i in range(N):
            y_pos = int(min(i * strip_h, body_h - 1))
            row_alpha = bird_body[y_pos, :, 3]
            nonzero = np.nonzero(row_alpha)[0]
            if len(nonzero) > 0:
                left_w = body_w/2 - nonzero[0]
                right_w = nonzero[-1] - body_w/2
            else:
                left_w = right_w = 0
                
            if i == 0:
                tangent = nodes[1] - nodes[0]
            elif i == N - 1:
                tangent = nodes[N-1] - nodes[N-2]
            else:
                tangent = nodes[i+1] - nodes[i-1]
                
            length = np.linalg.norm(tangent)
            if length > 0:
                tangent = tangent / length
            normal = np.array([-tangent[1], tangent[0]])
            
            p = nodes[i]
            left_points.append(p + normal * left_w)
            right_points.append(p - normal * right_w)

        for i in range(N-1):
            p1 = nodes[i]
            p2 = nodes[i+1]
            dist = np.linalg.norm(p2 - p1)
            angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) - 90
            
            strip = bird_body[i*strip_h:(i+1)*strip_h, :]
            
            if dist > 0:
                scale_y = dist / strip_h
                strip_resized = cv2.resize(strip, (body_w, int(dist) + 1))
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                frame = overlay_transparent(frame, strip_resized, mid_x, mid_y, angle=angle)
                
        boundary_pts = np.array(left_points + right_points[::-1], np.int32)
        cv2.polylines(frame, [boundary_pts], True, (0, 0, 0), 4, cv2.LINE_AA)
        
        current_head_img = bird_head_blink if is_blinking else bird_head
        p1, p2 = nodes[0], nodes[1]
        head_angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) - 90
        frame = overlay_transparent(frame, current_head_img, nodes[0][0], nodes[0][1], angle=head_angle*0.5)
        
        head_radius = 80
        cv2.circle(frame, (int(nodes[0][0]), int(nodes[0][1])), head_radius, (0, 0, 0), 4, cv2.LINE_AA)

        cv2.imshow('Bird Head', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
