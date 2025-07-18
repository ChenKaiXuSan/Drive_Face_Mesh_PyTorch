import cv2
import mediapipe as mp
from typing import Tuple, Optional
from pathlib import Path


# ---------- 模块初始化 ----------

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ---------- 图像处理相关 ----------

def load_image(img_path: Path) -> Tuple:
    """读取并转换为 RGB 图像"""
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")
    bgr_img = cv2.imread(str(img_path))
    if bgr_img is None:
        raise ValueError(f"cv2 failed to read image: {img_path}")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return bgr_img, rgb_img


def predict_face_mesh(image_rgb, model) -> Optional[list]:
    """执行 FaceMesh 预测"""
    results = model.process(image_rgb)
    return results.multi_face_landmarks


def draw_face_mesh(image_bgr, face_landmarks) -> any:
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


# ---------- 单张图片处理 ----------

def process_image_with_face_mesh(
    img_path: Path,
    output_path: Optional[Path] = None,
    show: bool = False,
    max_faces: int = 1
):
    """处理单张图片，绘制人脸 mesh"""
    bgr_img, rgb_img = load_image(img_path)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        refine_landmarks=True,
    ) as model:
        landmarks = predict_face_mesh(rgb_img, model)

        annotated_img = draw_face_mesh(bgr_img, landmarks) if landmarks else bgr_img

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated_img)
            print(f"[✓] Saved output to {output_path}")

        if show:
            cv2.imshow("Face Mesh (Image)", annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated_img


# ---------- 视频处理 ----------

def process_video_with_face_mesh(
    video_path: Path,
    output_path: Optional[Path] = None,
    show: bool = False,
    max_faces: int = 1
):
    """逐帧处理视频中的人脸 mesh，可选保存输出视频"""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    writer = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        refine_landmarks=True,
    ) as model:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = predict_face_mesh(rgb_frame, model)

            if landmarks:
                frame = draw_face_mesh(frame, landmarks)

            if show:
                cv2.imshow("Face Mesh (Video)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if writer:
                writer.write(frame)

    cap.release()
    if writer:
        writer.release()
        print(f"[✓] Saved video output to {output_path}")
    if show:
        cv2.destroyAllWindows()


# ---------- 调试入口 ----------

if __name__ == "__main__":
    # 示例用法（调试时可注释切换）
    # process_image_with_face_mesh(Path("example.jpg"), output_path=Path("out.jpg"), show=True)
    # process_video_with_face_mesh(Path("input.mp4"), output_path=Path("out.mp4"), show=True)
    pass