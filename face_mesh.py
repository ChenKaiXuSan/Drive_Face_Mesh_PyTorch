import cv2
import mediapipe as mp

# 初始化 mediapipe 模块（只初始化一次）
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh_model = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True
)


def load_image(img_path: str):
    """读取并转换为 RGB 图像"""
    bgr_img = cv2.imread(img_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return bgr_img, rgb_img


def predict_face_mesh(image_rgb):
    """执行 FaceMesh 预测"""
    results = face_mesh_model.process(image_rgb)
    return results.multi_face_landmarks


def draw_face_mesh(image_bgr, face_landmarks):
    """在图像上绘制 mesh（contours + tessellation）"""
    for landmarks in face_landmarks:
        mp_drawing.draw_landmarks(
            image=image_bgr,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            image=image_bgr,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
    return image_bgr


def process_video_with_face_mesh(
    video_path: str, output_path: str = None, show: bool = False
):
    """逐帧处理视频中的人脸 mesh，可选保存输出视频"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    # 视频保存设置（可选）
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = predict_face_mesh(rgb_frame)

        if landmarks:
            frame = draw_face_mesh(frame, landmarks)

        if show:
            cv2.imshow("Face Mesh (Video)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
                break

        if writer:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
