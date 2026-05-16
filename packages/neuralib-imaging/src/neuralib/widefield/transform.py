# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportIncompatibleMethodOverride=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportPossiblyUnboundVariable=false, reportAssignmentType=false

import colorsys
import traceback
from datetime import datetime
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
from argclz import AbstractParser
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont, QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from skimage.io import imread

from .ccf import DorsalCCF

__all__ = ['RegistrationApp', 'RegistrationOptions']


# TODO might load reference from retinotopic and align with cur widefield?
# TODO cur widefield apply translation or rotation together with projective2d, and save as 2d

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    arr_min = arr.min()
    arr_range = np.ptp(arr)
    if arr_range == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr8 = 255 * (arr - arr_min) / arr_range
    return np.ascontiguousarray(arr8.astype(np.uint8))


def _label_image_to_rgb(arr: np.ndarray, label_colors: dict[int, tuple[int, int, int]] | None = None) -> np.ndarray:
    rgb_img = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for label in np.unique(arr):
        if label == 0:
            continue

        label_int = int(label)
        if label_colors is not None and label_int in label_colors:
            color = label_colors[label_int]
        else:
            color = tuple(int(c * 255) for c in plt.cm.tab20(label_int % 20)[:3])

        rgb_img[arr == label] = color

    return np.ascontiguousarray(rgb_img)


def _distinct_label_colors(labels: list[int]) -> dict[int, tuple[int, int, int]]:
    colors = {}
    for index, label in enumerate(sorted(labels)):
        hue = (index * 0.618033988749895) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.70, 0.95)
        colors[label] = (int(red * 255), int(green * 255), int(blue * 255))
    return colors


def np_to_qpixmap(
        arr: np.ndarray,
        colorize_labels: bool = False,
        label_colors: dict[int, tuple[int, int, int]] | None = None
) -> QPixmap:
    try:
        match arr.ndim:
            case 2:
                if colorize_labels:
                    rgb_img = _label_image_to_rgb(arr, label_colors)
                else:
                    # Grayscale
                    arr8 = _normalize_to_uint8(arr)
                    h, w = arr8.shape
                    image = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8)
                    return QPixmap.fromImage(image)

            case 3:
                if arr.shape[2] == 3:  # RGB image
                    rgb_img = _normalize_to_uint8(arr)
                else:
                    # video stack
                    frame = arr[0]  # Take first frame
                    arr8 = _normalize_to_uint8(frame)
                    h, w = arr8.shape
                    image = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8)
                    return QPixmap.fromImage(image)

            case _:
                raise ValueError(f"Unsupported array shape: {arr.shape}")

        h, w = rgb_img.shape[:2]
        image = QImage(rgb_img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image)

    except Exception as e:
        print("Error in np_to_qpixmap:", e)
        return QPixmap()


class ZoomPanGraphicsView(QGraphicsView):
    def __init__(self, scene, drop_callback=None):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(300, 300)
        self.setAcceptDrops(True)
        self.drop_callback = drop_callback

    def wheelEvent(self, event: QWheelEvent):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(zoom_factor, zoom_factor)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.npy', '.mp4', '.avi', '.mov')):
                    if self.drop_callback:
                        self.drop_callback(file_path)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)

        # get the scene coordinates of the viewport corners
        scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        left, top, right, bottom = scene_rect.left(), scene_rect.top(), scene_rect.right(), scene_rect.bottom()

        pen = QPen(Qt.GlobalColor.white)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setFont(QFont("Sans", 8))

        # draw X axis line
        painter.drawLine(QPointF(left, bottom), QPointF(right, bottom))
        # draw Y axis line
        painter.drawLine(QPointF(left, top), QPointF(left, bottom))

        # draw X ticks + labels every 50 pixels
        start_x = np.floor(left / 50) * 50
        for x in np.arange(start_x, right, 50):
            tick_pos = QPointF(x, bottom)
            painter.drawLine(tick_pos, tick_pos + QPointF(0, -5))
            painter.drawText(tick_pos + QPointF(2, -7), f"{int(x)}")

        # draw Y ticks + labels every 50 pixels
        start_y = np.floor(top / 50) * 50
        for y in np.arange(start_y, bottom, 50):
            tick_pos = QPointF(left, y)
            painter.drawLine(tick_pos, tick_pos + QPointF(5, 0))
            painter.drawText(tick_pos + QPointF(7, 3), f"{int(y)}")


class ImageScene(QGraphicsScene):
    def __init__(
            self,
            name,
            click_callback,
            parent=None,
            colorize_labels=False,
            label_colors=None,
            show_boundaries=False
    ):
        super().__init__(parent)
        self.name = name
        self.pix_item = None
        self.click_callback = click_callback
        self.colorize_labels = colorize_labels
        self.label_colors = label_colors
        self.show_boundaries = show_boundaries
        self.enabled = False
        self.points = []

    def set_image(self, array):
        try:
            display_array = self._display_array(array)
            self.clear()
            pixmap = np_to_qpixmap(
                display_array,
                colorize_labels=self.colorize_labels,
                label_colors=self.label_colors
            )
            self.pix_item = QGraphicsPixmapItem(pixmap)
            self.addItem(self.pix_item)
            for pt in self.points:
                self.addEllipse(pt[0] - 2, pt[1] - 2, 4, 4, brush=Qt.GlobalColor.red)
        except Exception as e:
            print(f"Error setting image in {self.name}:", e)

    def _display_array(self, array):
        if self.show_boundaries and self.colorize_labels:
            from skimage.segmentation import find_boundaries

            rgb_img = _label_image_to_rgb(array, self.label_colors)
            boundary = find_boundaries(array, mode='outer')
            rgb_img[boundary] = (255, 255, 255)
            return rgb_img

        return array

    def clear_scene(self):
        self.clear()
        self.pix_item = None
        self.points.clear()

    def draw_point(self, point):
        self.points.append(point)
        self.addEllipse(point[0] - 2, point[1] - 2, 5, 5, brush=Qt.GlobalColor.red)

    def mousePressEvent(self, event: QMouseEvent):
        if self.enabled and event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            self.draw_point([pos.x(), pos.y()])
            self.click_callback(self.name, [pos.x(), pos.y()])


class RegistrationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dorsal Map Registration Tool")

        # --- State ---
        self.point_pairs = []
        self.expecting = 'Wfield'

        # --- Scenes & Views ---
        self.wf_scene = ImageScene("Wfield", self.on_point_clicked)
        self.dorsal_scene = ImageScene(
            "Dorsal",
            self.on_point_clicked,
            colorize_labels=True,
            show_boundaries=True
        )
        self.wf_view = ZoomPanGraphicsView(self.wf_scene)
        self.dorsal_view = ZoomPanGraphicsView(self.dorsal_scene)

        # --- Status Labels ---
        self.hist_shape_label = QLabel("WF size: N/A")
        self.dorsal_shape_label = QLabel("Dorsal size: N/A")

        # --- Controls ---
        # File I/O
        self.load_hist_btn = QPushButton("Load Widefield Image (Image/Movie)")
        self.load_dorsal_btn = QPushButton("Load Dorsal Map (.npy)")

        # Video controls
        self.play_btn = QPushButton("▶ Play")
        self.pause_btn = QPushButton("⏸ Pause")

        # Transform controls
        self.transform_btn = QPushButton("Apply Transform")
        self.save_tf_btn = QPushButton("Save Transform")
        self.save_img_btn = QPushButton("Save Image")
        self.transform_type_box = QComboBox()
        self.transform_type_box.addItems(["projective", "similarity"])
        self.ransac_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.ransac_threshold_slider.setRange(10, 100)
        self.ransac_threshold_slider.setValue(50)
        self.ransac_threshold_label = QLabel("RANSAC Threshold: 50 px")
        self.ransac_threshold_label.hide()
        self.ransac_threshold_slider.hide()

        # Annotation controls
        self.clear_btn = QPushButton("Clear All Points")
        self.undo_btn = QPushButton("Undo Last Pair")

        # Region selection
        self.update_dorsal_btn = QPushButton("Update Dorsal View")
        self.region_list = QListWidget()
        self.region_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.region_list.setMinimumHeight(260)
        self.region_list.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Hemisphere selection
        self.left_hemisphere_checkbox = QCheckBox("Left Hemisphere")
        self.right_hemisphere_checkbox = QCheckBox("Right Hemisphere")

        # Reshape and View
        self.resize_btn = QPushButton("Resize Both Images")
        self.resize_width = QSpinBox()
        self.resize_width.setRange(64, 2048)
        self.resize_width.setValue(300)

        self.resize_height = QSpinBox()
        self.resize_height.setRange(64, 2048)
        self.resize_height.setValue(300)

        self.zoom_to_fit_btn = QPushButton("Zoom to Fit Views")

        # Log console
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: black; color: lime; font-family: monospace;")

        # === Group Boxes ===
        # File I/O Group
        io_group = QGroupBox("File I/O")
        io_layout = QVBoxLayout(io_group)
        io_layout.addWidget(self.load_hist_btn)
        io_layout.addWidget(self.load_dorsal_btn)

        # Video Controls Group
        video_group = QGroupBox("Video Controls")
        video_layout = QHBoxLayout(video_group)
        video_layout.addWidget(self.play_btn)
        video_layout.addWidget(self.pause_btn)

        # Transform Controls Group
        transform_group = QGroupBox("Transform Controls")
        transform_layout = QVBoxLayout(transform_group)
        transform_layout.addWidget(QLabel("Transform type:"))
        transform_layout.addWidget(self.transform_type_box)
        transform_layout.addWidget(self.transform_btn)
        transform_layout.addWidget(self.save_tf_btn)
        transform_layout.addWidget(self.save_img_btn)
        transform_layout.addWidget(self.ransac_threshold_label)
        transform_layout.addWidget(self.ransac_threshold_slider)

        # Annotation Group
        points_group = QGroupBox("Point Annotation")
        points_layout = QHBoxLayout(points_group)
        points_layout.addWidget(self.clear_btn)
        points_layout.addWidget(self.undo_btn)

        # Region Selection Group
        region_group = QGroupBox("Region Selection")
        region_layout = QVBoxLayout(region_group)
        region_layout.addWidget(QLabel("Select region(s):"))
        region_layout.addWidget(self.region_list)
        region_layout.addWidget(self.update_dorsal_btn)

        # Hemisphere selection sub-layout
        hemisphere_layout = QHBoxLayout()
        hemisphere_layout.addWidget(self.left_hemisphere_checkbox)
        hemisphere_layout.addWidget(self.right_hemisphere_checkbox)
        region_layout.addLayout(hemisphere_layout)

        region_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Resize group
        resize_group = QGroupBox("Resize Widefield and Map")
        resize_layout = QHBoxLayout(resize_group)
        resize_layout.addWidget(QLabel("Width:"))
        resize_layout.addWidget(self.resize_width)
        resize_layout.addWidget(QLabel("Height:"))
        resize_layout.addWidget(self.resize_height)
        resize_layout.addWidget(self.resize_btn)

        view_group = QGroupBox("View Tools")
        view_layout = QVBoxLayout(view_group)
        view_layout.addWidget(self.zoom_to_fit_btn)
        view_layout.addStretch(1)

        # --- Assemble Top Controls ---
        controls_layout = QGridLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(10)

        controls_layout.addWidget(io_group, 0, 0)
        controls_layout.addWidget(video_group, 0, 1)
        controls_layout.addWidget(region_group, 0, 2, 4, 1)

        controls_layout.addWidget(transform_group, 1, 0, 1, 2)
        controls_layout.addWidget(points_group, 2, 0)
        controls_layout.addWidget(view_group, 2, 1)
        controls_layout.addWidget(resize_group, 3, 0, 1, 2)

        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setColumnStretch(2, 2)
        controls_layout.setRowStretch(3, 1)

        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # --- Main area: status bar, views, log ---
        # Status bar
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.hist_shape_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.dorsal_shape_label)
        status_widget = QWidget()
        status_widget.setLayout(status_layout)

        # Views splitter
        view_splitter = QSplitter(Qt.Orientation.Horizontal)
        view_splitter.addWidget(self.wf_view)
        view_splitter.addWidget(self.dorsal_view)

        # Vertical splitter stacking status, views, and log
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(status_widget)
        main_splitter.addWidget(view_splitter)
        main_splitter.addWidget(self.log_box)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 5)
        main_splitter.setStretchFactor(2, 0)

        # Final layout: controls on top, content below
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        main_layout.addWidget(controls_widget)
        main_layout.addWidget(main_splitter, 1)
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # --- Signal-Slot Connections ---
        for btn, fn in [
            (self.load_hist_btn, self.load_widefield),
            (self.load_dorsal_btn, self.load_dorsal_map),
            (self.play_btn, self.play_video),
            (self.pause_btn, self.pause_video),
            (self.transform_btn, self.apply_transform),
            (self.save_tf_btn, self.save_transform),
            (self.save_img_btn, self.save_dorsal_image),
            (self.clear_btn, self.clear_all_points),
            (self.undo_btn, self.undo_last_pair),
            (self.zoom_to_fit_btn, self.zoom_to_fit_views),
            (self.update_dorsal_btn, self.update_dorsal_from_region),
            (self.resize_btn, self.resize_both_images)
        ]:
            btn.clicked.connect(self.safe_call(fn))

        # Checkbox connections for hemisphere selection
        self.left_hemisphere_checkbox.stateChanged.connect(self.safe_call(self.on_hemisphere_changed))
        self.right_hemisphere_checkbox.stateChanged.connect(self.safe_call(self.on_hemisphere_changed))

        self.ransac_threshold_slider.valueChanged.connect(self.update_ransac_threshold)

        # --- Finish initialization ---
        self.video_path = None
        self.wf_img = None
        self.dorsal_map = None
        self.current_transform = None

        # video
        self.video_cap = None
        self.current_frame_idx = None
        self.tiff_frames = None
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.safe_call(self.next_frame))

        # Live transform overlay attributes
        self.transform_stats = None
        self.live_fig = None
        self.live_ax = None
        self.live_overlay = None
        self.live_frame_text = None
        self.original_next_tiff_frame = None
        self.original_next_frame = None

        # Method references (will be reassigned during live overlay)
        self._next_tiff_frame = self._next_tiff_frame
        self._next_frame = self._next_frame

        self.ccf = DorsalCCF.from_json()
        self.dorsal_scene.label_colors = _distinct_label_colors([
            label.label
            for label in self.ccf.region_labels
        ])
        self.region_list.addItem("[All Regions]")
        for region in self.ccf.region_list:
            self.region_list.addItem(region)

    def log(self, message, level: Literal['info', 'warn', 'error', 'success', 'debug'] = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")

        colors = {
            "info": "lime",
            "warn": "yellow",
            "error": "red",
            "success": "cyan",
            "debug": "gray"
        }

        color = colors.get(level, "lime")
        level_tag = level.upper()
        formatted_msg = f'<span style="color: {color};">[{timestamp}] [{level_tag}] {message}</span>'
        self.log_box.append(formatted_msg)

    def safe_call(self, func):
        def wrapper():
            try:
                func()
            except Exception as e:
                self.log(f"Error: {e}\n{traceback.format_exc()}", "error")

        return wrapper

    def zoom_to_fit_views(self):
        if self.wf_scene.pix_item:
            self.wf_view.fitInView(self.wf_scene.pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        if self.dorsal_scene.pix_item:
            self.dorsal_view.fitInView(self.dorsal_scene.pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.log("Zoomed to fit both views.")

    # ============== #
    # Video Features #
    # ============== #

    def next_frame(self):
        if self.video_cap is None:
            return

        if self.video_cap == "tiff":
            # Safety check: ensure _next_tiff_frame is callable
            if callable(self._next_tiff_frame):
                self._next_tiff_frame()
            else:
                # Fallback to the original method
                RegistrationApp._next_tiff_frame(self)
        else:
            # Safety check: ensure _next_frame is callable
            if callable(self._next_frame):
                self._next_frame()
            else:
                # Fallback to the original method
                RegistrationApp._next_frame(self)

    def _next_tiff_frame(self):
        if not hasattr(self, 'tiff_frames') or not hasattr(self, 'current_frame_idx'):
            self.video_timer.stop()
            self.log("TIFF video data not available.", "error")
            return

        frame = self.tiff_frames[self.current_frame_idx]
        self.current_frame_idx = (self.current_frame_idx + 1) % self.tiff_frames.shape[0]

        # resize the frame to match current settings?
        if hasattr(self, 'wf_img') and self.wf_img is not None:
            # Use the current resized dimensions
            target_h, target_w = self.wf_img.shape[:2]
            if frame.shape[:2] != (target_h, target_w):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        self.wf_img = frame
        self.wf_scene.set_image(self.wf_img)
        self.hist_shape_label.setText(
            f"TIFF video frame {self.current_frame_idx}/{self.tiff_frames.shape[0]}: {frame.shape[1]} x {frame.shape[0]}"
        )

    def _next_frame(self):
        ret, frame = self.video_cap.read()
        if not ret:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_cap.read()
            if not ret:
                self.video_timer.stop()
                self.log("Video playback error.", "error")
                return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Check if we should resize the frame to match current settings
        if hasattr(self, 'wf_img') and self.wf_img is not None:
            # Use the current resized dimensions
            target_h, target_w = self.wf_img.shape[:2]
            if frame.shape[:2] != (target_h, target_w):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        self.wf_img = frame
        self.wf_scene.set_image(self.wf_img)
        self.hist_shape_label.setText(f"Histology video frame size: {frame.shape[1]} x {frame.shape[0]}")

    def play_video(self):
        if self.video_cap is not None:
            self.video_timer.start(30)
            self.log("Playing video.")

    def pause_video(self):
        self.video_timer.stop()
        self.log("Paused video.")

    # ============================== #
    # Load image/video OR dorsal map #
    # ============================== #

    def load_widefield(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Widefield Image or Movie")
        if not path:
            return

        if path.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
            self.video_path = path
            self.video_cap = cv2.VideoCapture(path)
            self.next_frame()
            self.log(f"Loaded movie: {path}", "success")
        else:
            loaded_data = imread(path)

            # Check if this is a multi-frame TIFF (3D array with many frames or 4D array)
            if loaded_data.ndim == 4 or (loaded_data.ndim == 3 and loaded_data.shape[0] > 10):
                self.video_path = path
                self.tiff_frames = loaded_data
                self.current_frame_idx = 0
                self.video_cap = "tiff"  # flag for TIFF video
                self.wf_img = loaded_data[0]  # show first frame
                self.wf_scene.set_image(self.wf_img)
                h, w = self.wf_img.shape[:2]
                self.hist_shape_label.setText(f"TIFF video size: {w} x {h} ({loaded_data.shape[0]} frames)")
                self.log(f"Loaded TIFF video: {path} ({loaded_data.shape[0]} frames)", "success")
            else:
                self.wf_img = loaded_data
                self.wf_scene.set_image(self.wf_img)
                self.video_cap = None
                h, w = self.wf_img.shape[:2]
                self.hist_shape_label.setText(f"widefield image size: {w} x {h}")
                self.log(f"Loaded image: {path}", "success")

            # Set max resize range to WF image size
            self.resize_width.setMaximum(w)
            self.resize_height.setMaximum(h)

        self.point_pairs.clear()
        self.expecting = 'Wfield'
        self.wf_scene.enabled = True
        self.dorsal_scene.enabled = False

    def load_dorsal_map(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Dorsal Map (.npy)")
        if not path:
            return
        arr = np.load(path)
        # if histology is already loaded, resize dorsal to match its pixel shape
        if self.wf_img is not None:
            h, w = self.wf_img.shape[:2]
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
            self.log(f'resize dorsal map to {w, h}', "debug")

        self.dorsal_map = arr
        self.dorsal_scene.set_image(self.dorsal_map)

        self.point_pairs.clear()
        self.expecting = 'Wfield'
        self.wf_scene.enabled = True
        self.dorsal_scene.enabled = False

        h, w = self.dorsal_map.shape[:2]
        self.dorsal_shape_label.setText(f"Dorsal map size: {w} x {h}")
        self.log(f"Loaded dorsal map: {path}", "success")

    def update_dorsal_from_region(self, crop: bool = False):
        if self.wf_img is None:
            self.log("Load the wfield image first!", level='warn')
            return

        # Start with region selection
        selected = [it.text() for it in self.region_list.selectedItems()
                    if it.text() != "[All Regions]"]
        if selected:
            ccf = self.ccf.select_region(selected)
            region_label = ", ".join(selected)
        else:
            ccf = self.ccf
            region_label = "[All Regions]"

        # Apply hemisphere selection
        ccf = self._apply_hemisphere_selection(ccf)

        # Build the display label
        left_checked = self.left_hemisphere_checkbox.isChecked()
        right_checked = self.right_hemisphere_checkbox.isChecked()
        if left_checked and not right_checked:
            hemisphere_label = " (Left Hemisphere)"
        elif right_checked and not left_checked:
            hemisphere_label = " (Right Hemisphere)"
        else:
            hemisphere_label = ""

        full_label = region_label + hemisphere_label

        # get the array and crop to nonzero if needed
        arr = ccf.to_numpy()
        if crop:
            mask = arr > 0
            if mask.any():
                ys, xs = np.where(mask)
                arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        # now resize to match histology (if loaded)
        if self.wf_img is not None:
            h, w = self.wf_img.shape[:2]
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
            self.log(f"Resized dorsal map to histology dims: {w}×{h}", "debug")

        # update the view and label
        self.dorsal_map = arr
        self.dorsal_scene.set_image(arr)
        self.dorsal_shape_label.setText(f"Dorsal size: {w} × {h}")
        self.log(f"Dorsal view updated for: {full_label}", "success")

    def resize_both_images(self):
        target_w = self.resize_width.value()
        target_h = self.resize_height.value()
        size = (target_w, target_h)
        self.log(f"Resizing both images to {size}")

        # Resize histology
        if self.wf_img is not None:
            hist = cv2.resize(self.wf_img, size, interpolation=cv2.INTER_LINEAR)
            self.wf_img = hist
            self.wf_scene.set_image(hist)
            self.hist_shape_label.setText(f"Wfield size: {hist.shape[1]} × {hist.shape[0]}")
        else:
            self.log("Wfield image not loaded.", "warn")

        # Resize dorsal
        if self.dorsal_map is not None:
            dorsal = cv2.resize(self.dorsal_map, size, interpolation=cv2.INTER_NEAREST)
            self.dorsal_map = dorsal
            self.dorsal_scene.set_image(dorsal)
            self.dorsal_shape_label.setText(f"Dorsal size: {dorsal.shape[1]} × {dorsal.shape[0]}")
        else:
            self.log("Dorsal map not loaded.", "warn")

        self.log("Resize complete.", "success")

    # ==================== #
    # Hemisphere Selection #
    # ==================== #

    def on_hemisphere_changed(self):
        """Handle hemisphere checkbox changes and update dorsal view"""
        # Automatically trigger dorsal view update when hemisphere selection changes
        self.update_dorsal_from_region()

    def _apply_hemisphere_selection(self, ccf):
        """Apply hemisphere selection to a CCF object based on checkbox states"""
        left_checked = self.left_hemisphere_checkbox.isChecked()
        right_checked = self.right_hemisphere_checkbox.isChecked()

        # Handle hemisphere selection logic
        if left_checked and right_checked:
            # Both checked: return full CCF (no hemisphere filtering)
            return ccf
        elif left_checked:
            # Only left checked: select left hemisphere
            return ccf.select_hemisphere('left')
        elif right_checked:
            # Only right checked: select right hemisphere
            return ccf.select_hemisphere('right')
        else:
            # Neither checked: return full CCF (no hemisphere filtering)
            return ccf

    def convert_to_boundary(self):
        """Convert current dorsal map to boundary using skimage boundary detection"""
        if self.dorsal_map is None:
            self.log("No dorsal map loaded. Please load or generate a dorsal map first.", "warn")
            return

        try:
            from skimage.segmentation import find_boundaries

            boundary = find_boundaries(self.dorsal_map, mode='outer')
            boundary_arr = np.where(boundary, self.dorsal_map, 0).astype(self.dorsal_map.dtype)

            self.dorsal_map = boundary_arr
            self.dorsal_scene.set_image(self.dorsal_map)

            h, w = self.dorsal_map.shape[:2]
            self.dorsal_shape_label.setText(f"Boundary map size: {w} × {h}")
            self.log("Converted dorsal map to boundary", "success")

        except Exception as e:
            self.log(f"Error converting to boundary: {e}", "error")

    # ================== #
    # Point / Annotation #
    # ================== #

    def on_point_clicked(self, image_name, point):
        if image_name != self.expecting:
            return
        if self.expecting == 'Wfield':
            self.point_pairs.append({'wfield': point, 'dorsal': None})
            self.expecting = 'Dorsal'
            self.wf_scene.enabled = False
            self.dorsal_scene.enabled = True
            self.log(f"WField point: {point}", "debug")
        else:
            self.point_pairs[-1]['dorsal'] = point
            self.expecting = 'Wfield'
            self.wf_scene.enabled = True
            self.dorsal_scene.enabled = False
            self.log(f"Dorsal point: {point}", "debug")

    def update_ransac_threshold(self):
        thr = self.ransac_threshold_slider.value()
        self.ransac_threshold_label.setText(f"RANSAC Threshold: {thr} px")

    def clear_all_points(self):
        self.point_pairs.clear()
        self.wf_scene.clear_scene()
        self.dorsal_scene.clear_scene()
        if self.wf_img is not None:
            self.wf_scene.set_image(self.wf_img)
        if self.dorsal_map is not None:
            self.dorsal_scene.set_image(self.dorsal_map)
        self.expecting = 'Wfield'
        self.wf_scene.enabled = True
        self.dorsal_scene.enabled = False
        self.log("Cleared all points and scenes.", "success")

    def undo_last_pair(self):
        if not self.point_pairs:
            return
        last_pair = self.point_pairs[-1]
        if last_pair['dorsal'] is None:
            self.expecting = 'Wfield'
        else:
            self.expecting = 'Wfield'
            self.point_pairs.pop()

        self.wf_scene.clear_scene()
        self.dorsal_scene.clear_scene()
        if self.wf_img is not None:
            self.wf_scene.set_image(self.wf_img)
        if self.dorsal_map is not None:
            self.dorsal_scene.set_image(self.dorsal_map)

        for pair in self.point_pairs:
            if pair['wfield'] is not None:
                self.wf_scene.draw_point(pair['wfield'])
            if pair['dorsal'] is not None:
                self.dorsal_scene.draw_point(pair['dorsal'])

        self.wf_scene.enabled = (self.expecting == 'Wfield')
        self.dorsal_scene.enabled = (self.expecting == 'Dorsal')
        self.log("Undid last point pair.", "success")

    # ================== #
    # Transform / Result #
    # ================== #

    def apply_transform(self):
        # gather valid click-pairs
        wfield_pts = [p['wfield'] for p in self.point_pairs if p['dorsal'] is not None]
        dorsal_pts = [p['dorsal'] for p in self.point_pairs if p['dorsal'] is not None]

        mode = self.transform_type_box.currentText()
        required_pts = 4 if mode == 'projective' else 2
        if len(wfield_pts) < required_pts:
            self.log(f"Need at least {required_pts} point pairs for {mode} transform.", "warn")
            return

        # Remember video playback state
        was_playing = self.video_timer.isActive()

        # assemble Nx2 float32 arrays in (x,y) order
        src_pts = np.array(wfield_pts, dtype=np.float32)
        dst_pts = np.array(dorsal_pts, dtype=np.float32)

        self.log(f"Applying '{mode}'…")

        if mode == 'projective':
            self.log(" • findHomography (DLT)", "debug")
            H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
            inliers = np.ones(len(src_pts), dtype=bool)
        elif mode == 'similarity':
            self.log(" • estimateAffinePartial2D (similarity, LMEDS)", "debug")
            A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
            if A is None:
                self.log("Similarity estimation failed.", "error")
                return
            H = np.vstack([A, [0.0, 0.0, 1.0]])
            inliers = (mask.ravel() == 1) if mask is not None else np.ones(len(src_pts), dtype=bool)
        else:
            self.log(f"Unsupported transform mode: {mode}", "error")
            return

        if H is None:
            self.log("Transform estimation failed.", "error")
            return

        # warp with OpenCV
        h, w = self.dorsal_map.shape[:2]
        if self.wf_img.dtype == np.uint8:
            img_uint8 = self.wf_img
        else:
            # normalize to 0-255 range regardless of original data type/range
            img_uint8 = cv2.normalize(self.wf_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        warped = cv2.warpPerspective(
            img_uint8, H, (w, h), flags=cv2.INTER_LINEAR
        )

        self.current_transform = H

        # compute & log RMSE on inliers only
        src_w = cv2.perspectiveTransform(src_pts[None, :, :], H)[0]
        errs = np.linalg.norm(src_w - dst_pts, axis=1)
        rmse = np.sqrt(np.mean(errs[inliers] ** 2)) if inliers.any() else np.nan
        self.log(f"RMSE: {rmse:.2f}px ({inliers.sum()}/{len(src_pts)} inliers)", "success")

        # Store transform for live video overlay
        self.current_transform = H
        self.transform_stats = {
            'mode': mode,
            'rmse': rmse,
            'inliers': dst_pts[inliers],
            'outliers': dst_pts[~inliers] if (~inliers).any() else np.array([]).reshape(0, 2)
        }

        # Start live video overlay plot if video is available
        if self.video_cap is not None:
            self.start_live_transform_plot()
            if was_playing:
                self.video_timer.start(30)
                self.log("Video playback continued with live transform overlay.", "debug")
        else:
            # Static diagnostic plot for single images
            from skimage.segmentation import find_boundaries

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(warped, cmap='gray', alpha=1.0)
            boundaries = find_boundaries(self.dorsal_map, mode='outer')
            boundary_overlay = np.ma.masked_where(~boundaries, self.dorsal_map)
            ax.imshow(boundary_overlay, cmap='tab20', alpha=0.8)

            # Plot point correspondences
            ax.plot(dst_pts[inliers, 0], dst_pts[inliers, 1], 'go', markersize=6, label='Inliers')
            ax.plot(dst_pts[~inliers, 0], dst_pts[~inliers, 1], 'ro', markersize=6, label='Outliers')
            ax.legend(loc='upper right')
            ax.set_title(f"{mode} Transform | RMSE={rmse:.1f}px")
            ax.axis('off')
            plt.show(block=False)

    def save_transform(self):
        """save transform matrix"""
        if self.current_transform is None:
            self.log("No transform to save.", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Transform", filter="NumPy (*.npy)")
        if path:
            np.save(path, self.current_transform)
            self.log(f"Transform saved to {path}", "success")

    def save_dorsal_image(self):
        if self.dorsal_map is None:
            self.log("No dorsal map to save.", "warn")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Dorsal Map", filter="NumPy array (*.npy)")
        if out_path:
            np.save(out_path, self.dorsal_map)
            self.log(f"Saved dorsal image: {out_path}", "success")

    # ========================= #
    # Live Transform Overlay    #
    # ========================= #

    def start_live_transform_plot(self):
        """Start a live matplotlib plot showing transformed video overlaid on dorsal map"""
        try:
            # Create the figure and axis
            self.live_fig, self.live_ax = plt.subplots(figsize=(10, 8))
            self.live_ax.set_title(
                f"Live Transform Overlay | {self.transform_stats['mode']} | RMSE={self.transform_stats['rmse']:.1f}px")
            self.live_ax.axis('off')

            # Set up the background dorsal map
            dorsal_display = self.dorsal_map.astype(float)
            dorsal_display[dorsal_display == 0] = np.nan
            self.live_ax.imshow(dorsal_display, cmap='tab20', alpha=0.6, aspect='equal')

            # Plot inliers and outliers
            if len(self.transform_stats['inliers']) > 0:
                self.live_ax.plot(self.transform_stats['inliers'][:, 0], self.transform_stats['inliers'][:, 1], 'go',
                                  markersize=8, label='Inliers')
            if len(self.transform_stats['outliers']) > 0:
                self.live_ax.plot(self.transform_stats['outliers'][:, 0], self.transform_stats['outliers'][:, 1], 'ro',
                                  markersize=8, label='Outliers')

            # Initialize the image overlay (will be updated each frame)
            h, w = self.dorsal_map.shape[:2]
            self.live_overlay = self.live_ax.imshow(np.zeros((h, w)), cmap='gray', alpha=0.6, aspect='equal')

            # Add a frame counter text
            self.live_frame_text = self.live_ax.text(0.02, 0.98, '', transform=self.live_ax.transAxes,
                                                     fontsize=12, color='white', weight='bold',
                                                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

            plt.tight_layout()
            plt.show(block=False)
            plt.draw()

            # Store the original next_frame methods and replace with live overlay versions
            # Make sure we're storing the actual original methods, not already wrapped ones
            if not hasattr(self, 'original_next_tiff_frame'):
                self.original_next_tiff_frame = self._next_tiff_frame
            if not hasattr(self, 'original_next_frame'):
                self.original_next_frame = self._next_frame

            self._next_tiff_frame = self._next_tiff_frame_with_overlay
            self._next_frame = self._next_frame_with_overlay

            self.log("Live transform overlay started.", "success")

        except Exception as e:
            self.log(f"Error starting live transform overlay: {e}", "error")

    def _next_tiff_frame_with_overlay(self):
        """TIFF frame advance with live transform overlay"""
        # Safety check: if original method is None, fall back to built-in method
        if self.original_next_tiff_frame is not None:
            self.original_next_tiff_frame()
        else:
            # Fall back to calling the original method directly
            RegistrationApp._next_tiff_frame(self)
        self.update_live_overlay()

    def _next_frame_with_overlay(self):
        """Regular video frame advance with live transform overlay"""
        # Safety check: if original method is None, fall back to built-in method
        if self.original_next_frame is not None:
            self.original_next_frame()
        else:
            # Fall back to calling the original method directly
            RegistrationApp._next_frame(self)
        self.update_live_overlay()

    def update_live_overlay(self):
        """Update the live matplotlib overlay with current transformed frame"""
        try:
            if not hasattr(self, 'live_fig') or not hasattr(self, 'current_transform'):
                return

            # check if the plot window is still open
            if not plt.fignum_exists(self.live_fig.number):
                # restore original methods
                self.stop_live_transform_plot()
                return

            # Get current frame and transform it
            if self.wf_img is not None and self.current_transform is not None:
                # Use the exact same frame that's displayed in the main GUI
                current_frame = self.wf_img.copy()

                # Convert frame to uint8 for consistent display
                frame_min, frame_max = current_frame.min(), current_frame.max()

                if current_frame.dtype != np.uint8:
                    # Apply the same normalization as np_to_qpixmap function
                    if np.issubdtype(current_frame.dtype, np.integer):
                        if current_frame.dtype == np.uint16:
                            # Common case: uint16 data
                            if frame_max > 255:
                                img_uint8 = (current_frame.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
                            else:
                                img_uint8 = current_frame.astype(np.uint8)
                        else:
                            # Other integer types - use min/max normalization
                            if frame_max > frame_min:
                                norm = 255 * (current_frame.astype(np.float32) - frame_min) / (frame_max - frame_min)
                                img_uint8 = norm.astype(np.uint8)
                            else:
                                img_uint8 = np.full_like(current_frame, 128, dtype=np.uint8)
                    else:
                        # Float types
                        if frame_max <= 1.0 and frame_min >= 0.0:
                            img_uint8 = (current_frame * 255).astype(np.uint8)
                        else:
                            if frame_max > frame_min:
                                norm = 255 * (current_frame - frame_min) / (frame_max - frame_min)
                                img_uint8 = norm.astype(np.uint8)
                            else:
                                img_uint8 = np.full_like(current_frame, 128, dtype=np.uint8)
                else:
                    img_uint8 = current_frame

                # Apply transform with same settings as initial transform
                h, w = self.dorsal_map.shape[:2]
                warped = cv2.warpPerspective(
                    img_uint8, self.current_transform, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                # Convert back to float for consistent display with dorsal map
                warped_float = warped.astype(np.float32)

                # Update the overlay image with consistent intensity scaling
                self.live_overlay.set_data(warped_float)
                # Use fixed intensity range for consistency
                self.live_overlay.set_clim(vmin=0, vmax=255)

                # Update frame counter
                if hasattr(self, 'current_frame_idx') and self.current_frame_idx is not None:
                    if hasattr(self, 'tiff_frames'):
                        frame_text = f"Frame: {self.current_frame_idx}/{self.tiff_frames.shape[0]}"
                    else:
                        frame_text = f"Frame: {self.current_frame_idx}"
                    self.live_frame_text.set_text(frame_text)

                # Refresh the plot
                self.live_fig.canvas.draw_idle()
                self.live_fig.canvas.flush_events()

        except Exception as e:
            self.log(f"Error updating live overlay: {e}", "error")
            print(f"Live overlay error details: {e}")  # Additional debug info

    def stop_live_transform_plot(self):
        """Stop the live transform plot and restore original frame methods"""
        try:
            # Restore original methods
            if hasattr(self, 'original_next_tiff_frame'):
                self._next_tiff_frame = self.original_next_tiff_frame
                del self.original_next_tiff_frame

            if hasattr(self, 'original_next_frame'):
                self._next_frame = self.original_next_frame
                del self.original_next_frame

            # Clean up plot references
            if hasattr(self, 'live_fig'):
                del self.live_fig
            if hasattr(self, 'live_ax'):
                del self.live_ax
            if hasattr(self, 'live_overlay'):
                del self.live_overlay
            if hasattr(self, 'live_frame_text'):
                del self.live_frame_text

            self.log("Live transform overlay stopped.", "debug")

        except Exception as e:
            self.log(f"Error stopping live overlay: {e}", "error")


class RegistrationOptions(AbstractParser):
    DESCRIPTION = 'Register widefield images to a dorsal cortex map'

    def run(self):
        app = QApplication.instance()
        owns_app = app is None
        if app is None:
            app = QApplication([])

        window = RegistrationApp()
        window.show()

        if owns_app:
            app.exec()
